"""
Centralized configuration for the resource allocation framework.

All hardcoded parameters from the original codebase are extracted here
to provide a single source of truth and enable easy extension.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """File system paths for data, profiles, and logs."""

    project_root: Path = Path(__file__).resolve().parents[2]

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Default node list / dependency paths (may be overridden per-workflow)
    default_node_list: str = "data/React/node_list.json"
    default_node_dependency: str = "data/React/node_dependency.json"

    # Profile output paths
    attn_profile: str = "Profiler/attn_profile.json"
    interference_profile: str = "Profiler/interference_profile.json"
    mlp_profile: str = "Profiler/mlp_profile.json"
    increment_profile: str = "Profiler/increment.json"

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path against the project root."""
        return self.project_root / relative_path


# ---------------------------------------------------------------------------
# Model parameters (LLM architecture)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """LLM model architecture parameters."""

    num_layers: int = 32
    num_heads: int = 32
    head_size: int = 128
    num_kv_heads: int = 8

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_size

    @property
    def kv_hidden_size(self) -> int:
        return self.num_kv_heads * self.head_size


# ---------------------------------------------------------------------------
# GPU / Hardware configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareConfig:
    """GPU and hardware-level configuration."""

    dtype: str = "bfloat16"
    device: str = "cuda:2"
    total_num_gpus: int = 32
    total_num_blocks: int = 100_000
    block_size: int = 16


# ---------------------------------------------------------------------------
# Profiler configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for GPU kernel profiling."""

    warmup: int = 5
    repeat: int = 10

    # Attention profiling ranges
    decode_batch_sizes: Tuple[int, int, int] = (16, 256, 16)
    decode_seq_lengths: Tuple[int, int, int] = (128, 10240, 128)

    prefill_query_lengths: Tuple[int, int, int] = (16, 512, 16)
    prefill_seq_lengths: Tuple[int, int, int] = (128, 10240, 128)

    # Interference profiling ranges
    interference_small_lengths: Tuple[int, int, int] = (1000, 9000, 1000)
    interference_large_lengths: List[int] = field(default_factory=lambda: [
        16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000,
    ])
    interference_small_batch_range: Tuple[int, int, int] = (1, 64, 1)

    # Increment profiling
    increment_small_lengths_start: int = 1
    increment_small_lengths_range: Tuple[int, int, int] = (1000, 129000, 1000)
    increment_num_small_seqs_range: Tuple[int, int, int] = (4, 64, 4)


# ---------------------------------------------------------------------------
# Agent workflow configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    """Agent definitions and workflow-level parameters."""

    # Default agent list (will be overridden per workflow)
    agents: List[str] = field(default_factory=lambda: [
        "Reflector", "System", "Search",
    ])

    # KV cache tokens that can reside on a single GPU instance
    instance_cached_tokens: int = 216_000

    # Instance group constraints
    max_instance_group: int = 3
    min_instance_group: int = 1

    # Whether the agent supports KV cache reuse (prefix caching)
    reuse: bool = False

    # Whether to account for decode interference
    interference: bool = False


# ---------------------------------------------------------------------------
# Allocator configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AllocatorConfig:
    """Configuration for the resource allocation algorithm."""

    # Context range partitioning
    context_step: int = 2000
    max_context_length: int = 200_000

    # Dynamic programming
    dp_gpu_step: int = 1  # GPU granularity for DP
    gpus_per_instance: int = 1  # GPUs per vLLM instance (tensor parallelism size)

    # Throughput analysis
    chunk_size: int = 512
    cache_swap_latency: float = 0.1  # seconds

    # Convergence parameters for workflow allocation
    max_iterations: int = 100

    @property
    def max_context_idx(self) -> int:
        return self.max_context_length // self.context_step


# ---------------------------------------------------------------------------
# Workflow-level configuration (combines sub-configs for a specific run)
# ---------------------------------------------------------------------------
@dataclass
class WorkflowConfig:
    """Complete configuration for a single allocation run.

    Bundles all sub-configs and allows per-run overrides.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)

    # Per-run overrides
    workflow_name: str = "default"
    node_list_path: Optional[str] = None  # overrides paths.default_node_list
    node_dependency_path: Optional[str] = None

    def get_node_list_path(self) -> str:
        """Get the effective node list path for this run."""
        return self.node_list_path or self.paths.default_node_list

    def get_node_dependency_path(self) -> str:
        """Get the effective node dependency path for this run."""
        return self.node_dependency_path or self.paths.default_node_dependency


# ---------------------------------------------------------------------------
# Pre-built configurations for known workflows
# ---------------------------------------------------------------------------
def get_reflection_config() -> WorkflowConfig:
    """Configuration for the Reflection workflow (eigent dataset)."""
    return WorkflowConfig(
        workflow_name="reflection",
        node_list_path="data/Reflection/node_list.json",
        node_dependency_path="data/Reflection/node_dependency.json",
        agent=AgentConfig(
            agents=["Reflector", "System", "Search"],
            instance_cached_tokens=216_000,
            max_instance_group=3,
            min_instance_group=1,
            reuse=False,
            interference=False,
        ),
        hardware=HardwareConfig(total_num_gpus=32),
    )


def get_single_agent_config() -> WorkflowConfig:
    """Configuration for single-agent analysis (used in agent_partition.py)."""
    return WorkflowConfig(
        workflow_name="single_agent",
        agent=AgentConfig(
            agents=["Search"],
            instance_cached_tokens=216_000,
            max_instance_group=2,
            min_instance_group=1,
            reuse=True,
            interference=True,
        ),
        hardware=HardwareConfig(total_num_gpus=16),
    )
