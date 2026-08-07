# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A block manager that manages token blocks."""
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple
import time

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str


class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

        # [Instance] Cached entries kept in GPU memory.
        # Key:   (query_id, invocation_id)
        # Value: (request_id, seq_ids, token_ids)
        # Blocks are NOT freed — they stay in GPU memory for prefix reuse.
        self._gpu_cached_entries: Dict[Tuple[int, int],
                                        Tuple[str, List[int],
                                              List[int]]] = {}
        self.cached_dict: Dict[Tuple[int, int], str] = {}

        # [Instance] Cached entries kept in CPU memory.
        # Same structure as _gpu_cached_entries, but blocks have been swapped
        # to CPU memory (not freed) for later prefix reuse.
        self._cpu_cached_entries: Dict[Tuple[int, int],
                                        Tuple[str, List[int],
                                              List[int]]] = {}

        # [Instance] Query-level FCFS + within-query LRU tracking.
        # _query_timestamps:  query_id -> registration epoch.
        self._query_timestamps: Dict[int, float] = {}
        # _entry_access:       (query_id, invocation_id) → last_access epoch.
        #                     Updated on cache hit (LRU tracking).
        self._entry_access: Dict[Tuple[int, int], float] = {}

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash)

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def remove_seq_from_computed_blocks_tracker(self, seq: Sequence) -> None:
        seq_id = seq.seq_id
        self._computed_blocks_tracker.remove_seq(seq_id)

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.block_allocator.reset_prefix_cache(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    # [Instance] KV-cache retention in GPU memory  ----------------------------

    def retain_blocks(self, seq_group: SequenceGroup) -> None:
        """Register a finished seq_group's blocks as 'retained' in GPU memory.

        Called from :meth:`Scheduler._free_finished_seq_group` when reusable
        cache should remain in GPU memory.  The blocks are **not** freed; they
        stay in GPU memory so that future requests with the
        same ``(query_id, invocation_id)`` can reuse them via prefix caching.

        If an entry for the same key already exists it is evicted first.
        """
        sp = seq_group.sampling_params
        if sp is None or sp.extra_args is None:
            return
        query_id = sp.extra_args.get("query_id")
        invocation_id = sp.extra_args.get("invocation_id")
        if query_id is None or invocation_id is None:
            return

        key = (int(query_id), int(invocation_id))
        self.evict_cached_blocks(key)  # replace stale entry

        seq_ids: List[int] = []
        token_ids: List[int] = []
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                seq_ids.append(seq.seq_id)
                token_ids = seq.get_output_token_ids()

        self._gpu_cached_entries[key] = (seq_group.request_id, seq_ids, token_ids)
        self.cached_dict[key] = seq_group.request_id
        # Track access timestamps for FCFS+LRU eviction
        now = time.time()
        self._entry_access[key] = now
        if query_id not in self._query_timestamps:
            self._query_timestamps[query_id] = now

    # [Instance] CPU cache  ----------------------------------------------------

    def swap_out_finished(self, seq_group: SequenceGroup) -> bool:
        """Swap a finished seq_group's blocks from GPU to CPU memory.

        Called when reusable cache should remain in CPU memory. Blocks are
        moved to CPU (not freed) for later prefix-cache reuse.

        If CPU memory is exhausted, blocks are freed as a fallback.

        Returns True when at least one finished sequence was successfully
        swapped to CPU and can be registered as CPU-cached.
        """
        swapped_any = False
        for seq in seq_group.get_seqs():
            if seq.seq_id not in self.block_tables:
                continue
            blocks = self.block_tables[seq.seq_id].blocks
            if blocks is None or len(blocks) == 0:
                continue
            try:
                self.block_allocator.swap(
                    blocks=blocks,
                    src_device=Device.GPU,
                    dst_device=Device.CPU,
                )
                self.block_tables[seq.seq_id].update(blocks)
                swapped_any = True
            except Exception:
                # CPU OOM fallback: free the blocks
                for block in blocks:
                    self.block_allocator.free(block)
        return swapped_any

    def register_cpu_cache(self, seq_group: SequenceGroup) -> None:
        """Register a finished seq_group's blocks as CPU-cached.

        Called after :meth:`swap_out_finished`.  Records the mapping so that
        future requests with the same ``(query_id, invocation_id)`` can swap
        the blocks back from CPU to GPU.
        """
        sp = seq_group.sampling_params
        if sp is None or sp.extra_args is None:
            return
        query_id = sp.extra_args.get("query_id")
        invocation_id = sp.extra_args.get("invocation_id")
        if query_id is None or invocation_id is None:
            return

        key = (int(query_id), int(invocation_id))
        # Evict stale entry (both GPU and CPU)
        self.evict_cached_blocks(key)

        seq_ids: List[int] = []
        token_ids: List[int] = []
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                seq_ids.append(seq.seq_id)
                token_ids = seq.get_output_token_ids()

        self._cpu_cached_entries[key] = (seq_group.request_id, seq_ids, token_ids)
        # Track access timestamps for FCFS+LRU eviction
        now = time.time()
        self._entry_access[key] = now
        if query_id not in self._query_timestamps:
            self._query_timestamps[query_id] = now

    def swap_in_cached(self, seq_ids: List[int]) -> List[Tuple[int, int]]:
        """Swap CPU-cached blocks back to GPU for prefix reuse.

        Returns:
            List of (cpu_block_id, gpu_block_id) mappings.
        """
        physical_block_id_mapping: List[Tuple[int, int]] = []
        for seq_id in seq_ids:
            if seq_id not in self.block_tables:
                continue
            blocks = self.block_tables[seq_id].blocks
            if blocks is None or len(blocks) == 0:
                continue
            seq_swap_mapping = self.block_allocator.swap(
                blocks=blocks,
                src_device=Device.CPU,
                dst_device=Device.GPU,
            )
            self.block_tables[seq_id].update(blocks)
            seq_mapping = [
                (self.block_allocator.get_physical_block_id(Device.CPU, cpu_id),
                 self.block_allocator.get_physical_block_id(Device.GPU, gpu_id))
                for cpu_id, gpu_id in seq_swap_mapping.items()
            ]
            physical_block_id_mapping.extend(seq_mapping)
        return physical_block_id_mapping

    # [Instance] Unified cache lookup  -----------------------------------------

    def lookup_cache(self, query_id: int,
                     invocation_id: int) -> Optional[Tuple[str, List[int]]]:
        """Check GPU cached_dict.  (Backward-compatible)."""
        entry = self._gpu_cached_entries.get((query_id, invocation_id))
        if entry is not None:
            request_id, _seq_ids, token_ids = entry
            return (request_id, token_ids)
        return None

    def lookup_any_cache(
            self, query_id: int,
            invocation_id: int) -> Optional[Tuple[str, List[int], bool]]:
        """Check GPU + CPU caches.

        Returns:
            ``(request_id, token_ids, is_gpu)`` or ``None``.
            ``is_gpu=True`` means blocks are already in GPU memory.
        """
        key = (query_id, invocation_id)
        gpu_entry = self._gpu_cached_entries.get(key)
        if gpu_entry is not None:
            request_id, _seq_ids, token_ids = gpu_entry
            return (request_id, token_ids, True)
        cpu_entry = self._cpu_cached_entries.get(key)
        if cpu_entry is not None:
            request_id, _seq_ids, token_ids = cpu_entry
            return (request_id, token_ids, False)
        return None

    def consume_cached(
            self, query_id: int,
            invocation_id: int) -> Optional[Tuple[str, List[int], bool]]:
        """Consume a cached entry — swaps CPU→GPU if needed, then frees blocks.

        This implements project.md point 4:
        - GPU hit: blocks already in GPU → free old block_tables so the
          finished request is cleaned up after prefix reuse.
        - CPU hit: swap blocks CPU→GPU → free old block_tables.

        vLLM's prefix-caching allocator (called later in ``_schedule_prefills``)
        will find these blocks by hash and reuse them automatically.

        Returns:
            ``(request_id, token_ids, is_gpu)`` or ``None``.
        """
        key = (query_id, invocation_id)

        # Remove LRU tracking for consumed entry
        self._entry_access.pop(key, None)

        # Check GPU first
        gpu_entry = self._gpu_cached_entries.pop(key, None)
        if gpu_entry is not None:
            _request_id, seq_ids, token_ids = gpu_entry
            self.cached_dict.pop(key, None)
            self._free_cached_seq_blocks(seq_ids)
            self._maybe_cleanup_query(key[0])
            return (_request_id, token_ids, True)

        # Check CPU
        cpu_entry = self._cpu_cached_entries.pop(key, None)
        if cpu_entry is not None:
            _request_id, seq_ids, token_ids = cpu_entry
            self.swap_in_cached(seq_ids)       # CPU → GPU
            self._free_cached_seq_blocks(seq_ids)  # free old block_tables
            self._maybe_cleanup_query(key[0])
            return (_request_id, token_ids, False)
        return None

    def _free_cached_seq_blocks(self, seq_ids: List[int]) -> None:
        """Free :attr:`block_tables` for a list of cached seq_ids."""
        for seq_id in seq_ids:
            if seq_id not in self.block_tables:
                continue
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id,
                self.block_tables[seq_id].physical_block_ids)
            self._last_access_blocks_tracker.remove_seq(seq_id)
            self._computed_blocks_tracker.remove_seq(seq_id)
            self.block_tables[seq_id].free()
            del self.block_tables[seq_id]

    def free_request_cache(self, seq_group: SequenceGroup) -> None:
        """Release retained cache for a completed request key.

        When ``free_cache=True``, both GPU and CPU cached entries for
        ``(query_id, invocation_id)`` must be removed. The currently finished
        seq_group is freed by the scheduler after this method returns.
        """
        sp = seq_group.sampling_params
        if sp is None or sp.extra_args is None:
            return
        query_id = sp.extra_args.get("query_id")
        invocation_id = sp.extra_args.get("invocation_id")
        if query_id is None or invocation_id is None:
            return
        self.evict_cached_blocks((int(query_id), int(invocation_id)))

    def evict_cached_blocks(self, key: Tuple[int, int]) -> None:
        """Evict cached entries from both GPU and CPU, freeing their blocks."""
        # Clean up tracking
        self._entry_access.pop(key, None)

        gpu_entry = self._gpu_cached_entries.pop(key, None)
        if gpu_entry is not None:
            _request_id, seq_ids, _token_ids = gpu_entry
            self.cached_dict.pop(key, None)
            self._free_cached_seq_blocks(seq_ids)

        cpu_entry = self._cpu_cached_entries.pop(key, None)
        if cpu_entry is not None:
            _request_id, seq_ids, _token_ids = cpu_entry
            self._free_cached_seq_blocks(seq_ids)

        self.cached_dict.pop(key, None)
        self._maybe_cleanup_query(key[0])

    # [Instance] Query FCFS + within-query LRU helpers  -----------------------

    def _maybe_cleanup_query(self, query_id: int) -> None:
        """Remove query timestamp if this query has no more cached entries."""
        for (qid, _iid) in self._gpu_cached_entries:
            if qid == query_id:
                return
        for (qid, _iid) in self._cpu_cached_entries:
            if qid == query_id:
                return
        self._query_timestamps.pop(query_id, None)

    def _collect_query_entries(
        self, query_id: int, gpu_only: bool = False
    ) -> List[Tuple[Tuple[int, int], bool]]:
        """Return all cached entries for a query, sorted by LRU (oldest first).

        Returns:
            List of ((query_id, invocation_id), is_gpu) sorted by
            last_access_time ascending (LRU = oldest first for eviction).
        """
        entries: List[Tuple[Tuple[int, int], bool, float]] = []
        for key in self._gpu_cached_entries:
            if key[0] == query_id:
                ts = self._entry_access.get(key, 0)
                entries.append((key, True, ts))
        if not gpu_only:
            for key in self._cpu_cached_entries:
                if key[0] == query_id:
                    ts = self._entry_access.get(key, 0)
                    entries.append((key, False, ts))
        # Sort by timestamp ascending (LRU: oldest → evict first)
        entries.sort(key=lambda x: x[2])
        return [(key, is_gpu) for key, is_gpu, _ts in entries]

    def _demote_gpu_cache_to_cpu(self, key: Tuple[int, int]) -> None:
        """Move one retained GPU cache entry to the CPU cache pool.

        This is the memory-pressure path: the cache is evicted from GPU in
        query-FCFS / invocation-LRU order but remains reusable from CPU if CPU
        swap space is available.
        """
        gpu_entry = self._gpu_cached_entries.pop(key, None)
        if gpu_entry is None:
            return

        request_id, seq_ids, token_ids = gpu_entry
        self.cached_dict.pop(key, None)

        stale_cpu = self._cpu_cached_entries.pop(key, None)
        if stale_cpu is not None:
            _old_request_id, old_seq_ids, _old_token_ids = stale_cpu
            self._free_cached_seq_blocks(old_seq_ids)

        swapped_any = False
        try:
            for seq_id in seq_ids:
                if seq_id not in self.block_tables:
                    continue
                blocks = self.block_tables[seq_id].blocks
                if blocks is None or len(blocks) == 0:
                    continue
                self.block_allocator.swap(
                    blocks=blocks,
                    src_device=Device.GPU,
                    dst_device=Device.CPU,
                )
                self.block_tables[seq_id].update(blocks)
                swapped_any = True
        except Exception:
            swapped_any = False

        if swapped_any:
            self._cpu_cached_entries[key] = (request_id, seq_ids, token_ids)
            self._entry_access.setdefault(key, time.time())
            if key[0] not in self._query_timestamps:
                self._query_timestamps[key[0]] = self._entry_access[key]
        else:
            self._entry_access.pop(key, None)
            self._free_cached_seq_blocks(seq_ids)
            self._maybe_cleanup_query(key[0])

    def evict_by_fcfs_lru(self, target_blocks: int = 1) -> int:
        """Evict cached entries following query-FCFS + within-query LRU.

        Eviction order:
          1. Pick the smallest query_id.
          2. Within that query, evict the LRU invocation (oldest last-access).
          3. Repeat until *target_blocks* have been freed or no entries remain.

        Args:
            target_blocks: Minimum number of GPU blocks to free.

        Returns:
            Number of blocks actually freed.
        """
        freed = 0
        sorted_queries = sorted(self._query_timestamps.keys())

        for query_id in sorted_queries:
            if freed >= target_blocks:
                break
            entries = self._collect_query_entries(query_id, gpu_only=True)
            for key, is_gpu in entries:
                if freed >= target_blocks:
                    break
                # Count blocks before eviction
                before = self.block_allocator.get_num_free_blocks(Device.GPU)
                self._demote_gpu_cache_to_cpu(key)
                after = self.block_allocator.get_num_free_blocks(Device.GPU)
                freed += max(0, after - before)

        return freed

    # [Instance] Free finished blocks  ----------------------------------------

    def free_finished_blocks(self, seq_group: SequenceGroup) -> None:
        """Free blocks of a finished seq_group without caching.

        Evicts any existing GPU/CPU cache entry for this
        ``(query_id, invocation_id)`` so completed requests do not hold memory
        indefinitely.
        """
        sp = seq_group.sampling_params
        if sp is None or sp.extra_args is None:
            return
        query_id = sp.extra_args.get("query_id")
        invocation_id = sp.extra_args.get("invocation_id")
        if query_id is None or invocation_id is None:
            return

        key = (int(query_id), int(invocation_id))
        self.evict_cached_blocks(key)  # free any existing cached entry

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)
