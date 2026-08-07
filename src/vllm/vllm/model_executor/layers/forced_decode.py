# SPDX-License-Identifier: Apache-2.0
"""
Forced-decode logits processor for vLLM.

When provided with a list of ``output_tokens``, this processor forces the
sampler to generate those exact tokens in order, regardless of the model's
logit distribution.  This is used for deterministic replay in agentic
workflows where the output is already known (e.g., cached responses).
"""

from typing import List, Optional

import torch


class ForceDecodeLogitsProcessor:
    """A vLLM-compatible logits processor that forces specific output tokens.

    At each decode step this processor reads the next target token from
    ``output_tokens`` and gives that token the largest finite logit while
    suppressing all other logits.  This forces greedy sampling to always
    select the exact target token.

    When all forced tokens have been generated, an EOS token is forced so
    the sequence terminates cleanly.

    Usage (via extra_args)::

        params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
            ignore_eos=True,
            extra_args={
                "output_tokens": [1, 2, 3],
                "eos_token_id": 2,
            },
        )
        engine.add_request(request_id, prompt, params)
    """

    def __init__(self, output_tokens: List[int], eos_token_id: int = -1):
        self._output_tokens = output_tokens
        self._eos_token_id = eos_token_id
        self._step: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current decode step (0-based)."""
        return self._step

    @property
    def remaining_tokens(self) -> int:
        """Number of tokens remaining to force-generate."""
        return len(self._output_tokens) - self._step

    @property
    def is_done(self) -> bool:
        """Whether all output tokens have been generated."""
        return self._step >= len(self._output_tokens)

    # ------------------------------------------------------------------
    # vLLM LogitsProcessor protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        token_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Modify logits to force the next target token.

        Called by vLLM's sampler at each decoding step.
        Expected signature: ``(token_ids: List[int], logits: Tensor) -> Tensor``
        """
        if self._step >= len(self._output_tokens):
            # All forced tokens consumed — force EOS to stop cleanly
            if self._eos_token_id >= 0:
                finfo = torch.finfo(logits.dtype)
                logits[:] = finfo.min
                logits[self._eos_token_id] = finfo.max
            return logits

        target_token = self._output_tokens[self._step]
        self._step += 1

        finfo = torch.finfo(logits.dtype)
        logits[:] = finfo.min
        logits[target_token] = finfo.max
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clone(self) -> "ForceDecodeLogitsProcessor":
        """Create a fresh copy (e.g. for beam search)."""
        return ForceDecodeLogitsProcessor(
            list(self._output_tokens),
            eos_token_id=self._eos_token_id,
        )

    def reset(self) -> None:
        """Reset the step counter to start over."""
        self._step = 0

    def get_current_target(self) -> Optional[int]:
        """Get the current target token ID, or None if done."""
        if self._step < len(self._output_tokens):
            return self._output_tokens[self._step]
        return None
