from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import logging
import os
import json
workflow = 'camel'

def setup_logger(name, path, level=logging.INFO):
    """Create and return a logger with its own file handler."""
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.FileHandler(f"{path}/{name}.log", mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Each request has an DAG to guide its execution
# The invocation of external tools is implemented in Engine

class NodeStatus(Enum):
    WAITING = auto()
    EXECUTING = auto()
    FINISHED = auto()

class Node:
    """A single LLM invocation node in the agentic workflow DAG.

    Identification:
        Nodes are identified by the triple (query_id, invocation_id, repeat).
        - invocation_id:  Identifies the session (shared across repeats).
        - repeat:         Distinguishes nodes within the same session.

    Dispatch fields:
        agent_id:      Which agent this node belongs to.
        input_tokens:  Token IDs of the input prompt.
        output_tokens: Token IDs of the expected output (forced decoding).
        keep_cache_in_gpu:
                       Whether reusable KV cache should stay in GPU memory.
        free_cache:   Whether this node's KV cache will not be reused.
        status:        WAITING / EXECUTING / FINISHED.

    Legacy fields (optional, kept for backwards compat):
        cached_tokens, input_tokens, output_tokens,
        has_next_repeat, next_repeat_cached_tokens, next_repeat_context,
        tool_call_finish_time, start_time, end_time.
    """

    __slots__ = (
        # ---- Identification ----
        "agent_id",
        "invocation_id",
        "repeat",
        "status",
        # ---- Controller dispatch fields (set at runtime) ----
        "query_id",
        "input_tokens",
        "output_tokens",
        "keep_cache_in_gpu",
        "free_cache",
        # ---- Timing ----
        "start_time",
        "end_time",
        # ---- Token count estimates ----
        "cached_tokens",
        "num_input_tokens",
        "num_output_tokens",
        # ---- Legacy optional fields ----
        "has_next_repeat",
        "next_repeat_cached_tokens",
        "next_repeat_context",
        "tool_call_finish_time",
    )

    def __init__(
        self,
        agent_id: str,
        invocation_id: int,
        status: NodeStatus,
        repeat: int = 0,
        keep_cache_in_gpu: bool = False,
        free_cache: bool = True,
        query_id: int = 0,
        input_tokens: Optional[List[int]] = None,
        output_tokens: Optional[List[int]] = None,
        # -- Token count estimates --
        cached_tokens: int = 0,
        num_input_tokens: int = 0,
        num_output_tokens: int = 0,
        has_next_repeat: bool = False,
        next_repeat_cached_tokens: Optional[int] = None,
        next_repeat_context: Optional[int] = None,
        tool_call_finish_time: float = 0.0,
    ):
        # Identification
        self.agent_id = agent_id
        self.invocation_id = invocation_id
        self.repeat = repeat
        self.status = status
        # Dispatch
        self.query_id = query_id
        self.input_tokens = input_tokens or []
        self.output_tokens = output_tokens or []
        self.keep_cache_in_gpu = keep_cache_in_gpu
        self.free_cache = free_cache
        # Timing
        self.start_time = 0.0
        self.end_time: Optional[float] = None
        # Token count estimates
        self.cached_tokens = cached_tokens
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.has_next_repeat = has_next_repeat
        self.next_repeat_cached_tokens = next_repeat_cached_tokens
        self.next_repeat_context = next_repeat_context
        self.tool_call_finish_time = tool_call_finish_time

    def assign_num_tokens(self, input_tokens: int = 128, output_tokens: int = 128, cached_tokens: int = 128):
        self.num_input_tokens = input_tokens if input_tokens != 0 else 10
        self.num_output_tokens = output_tokens
        self.cached_tokens = cached_tokens

    def __str__(self):
        return (
            f"Node(agent={self.agent_id}, iid={self.invocation_id}:{self.repeat}, "
            f"qid={self.query_id}, "
            f"keep_cache_in_gpu={self.keep_cache_in_gpu}, "
            f"free_cache={self.free_cache}, "
            f"in={self.num_input_tokens}, out={self.num_output_tokens}, "
            f"cached={self.cached_tokens})"
        )

    def __repr__(self):
        return self.__str__()

class DAG:
    _logger: Optional[logging.Logger] = None

    def __init__(self, work_dir) -> None:
        self.workflow = []

        self.excuted_node = set()
        self.entry = None
        self.node_count = 0
        self.node_mapping: Dict[Tuple[int,int], Node] = {}  # key = (invocation_id, repeat)
        self.start_time = time.time()
        self.end_time = None

        if DAG._logger is None:
            DAG._logger = setup_logger("dag", work_dir)
        self.logger = DAG._logger

        self.end_to_end_latency = None
    
    def init_workflow_by_json(self, node_dependency: list, node_list: list):
        self.workflow = self.parse_by_json(node_dependency, node_list)

    def parse_by_json(self, node_dependency: dict, node_list: list):
        workflow = defaultdict(lambda: [])  # (invocation_id, repeat) -> [] dependency list

        def _count_tokens(value):
            if isinstance(value, int):
                return value
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], int):
                    return value[0]
                return len(value)
            return 0

        def _token_list(value):
            if isinstance(value, list):
                return value
            if isinstance(value, int):
                return [value]
            return []

        for record in node_dependency:
            child_inv_id = int(record.split(':')[0])
            child_repeat = int(record.split(':')[1])
            if node_dependency[record] == []:
                workflow[(child_inv_id, child_repeat)] = []
            else:
                for dep in node_dependency[record]:
                    parent_inv_id = int(dep.split(':')[0])
                    parent_repeat = int(dep.split(':')[1])
                    workflow[(child_inv_id, child_repeat)].append((parent_inv_id, parent_repeat))

        for record in node_list:
            inv_id = int(record['node_id'].split(':')[0])
            node_repeat = int(record['node_id'].split(':')[1])
            agent_name = record['agent_name']
            node = Node(
                agent_id=agent_name,
                invocation_id=inv_id,
                status=NodeStatus.WAITING,
                repeat=node_repeat,
                input_tokens=_token_list(record.get("input_tokens", [])),
                output_tokens=_token_list(record.get("output_tokens", [])),
            )

            node.assign_num_tokens(
                input_tokens=_count_tokens(record.get("input_tokens", 0)),
                output_tokens=_count_tokens(record.get("output_tokens", 0)),
                cached_tokens=_count_tokens(record.get("cached_tokens", 0)),
            )

            self.node_mapping[(inv_id, node_repeat)] = node

        for (inv_id, rpt) in self.node_mapping:
            node = self.node_mapping[(inv_id, rpt)]
            next_key = (inv_id, rpt + 1)
            direct_repeat = (
                next_key in self.node_mapping
                and node_dependency.get(f"{inv_id}:{rpt+1}") == [f"{inv_id}:{rpt}"]
            )
            current_total_tokens = node.num_input_tokens + node.num_output_tokens
            reusable = (
                direct_repeat
                and self.node_mapping[next_key].cached_tokens <= current_total_tokens
            )

            if reusable:
                next_node = self.node_mapping[next_key]
                node.has_next_repeat = True
                node.keep_cache_in_gpu = True
                node.free_cache = False
                node.next_repeat_cached_tokens = next_node.cached_tokens
                node.next_repeat_context = (
                    next_node.cached_tokens + next_node.num_input_tokens
                )
            else:
                node.has_next_repeat = False
                node.keep_cache_in_gpu = False
                node.free_cache = True

        self.node_count = len(self.node_mapping)
        return workflow
            
    def update_result(
        self,
        invocation_id: int,
        query_id:int,
        repeat:int
    ):   
        key = (invocation_id, repeat)
        self.logger.info(f"Node {invocation_id}:{repeat} in Query{query_id} completed. "
                         f"Time {time.time() - self.node_mapping[key].start_time:.2f}s")
        self.excuted_node.add(key)
        self.node_mapping[key].status = NodeStatus.FINISHED
        self.node_mapping[key].end_time = time.time()

        node_list = []
        for record in self.workflow:
            flag = True
            for dependency_tuple in self.workflow[record]:
                if dependency_tuple not in self.excuted_node:
                    flag = False

            if flag and self.node_mapping[record].status == NodeStatus.WAITING:
                self.node_mapping[record].status = NodeStatus.EXECUTING
                self.node_mapping[record].start_time = time.time()
                node_list.append(self.node_mapping[record])  
        
        if self.is_completed():
            self.end_time = time.time()
            self.end_to_end_latency = self.end_time-self.start_time

        return node_list            
                    
    def get_entry_nodes(self) -> List[Node]:
        """Return entry nodes with no dependencies, marking them EXECUTING.

        Entry nodes are those whose dependency list in workflow is empty.
        Used by the controller when first submitting a new query — no
        predecessor has executed, so we skip the normal update_result path.
        """
        node_list = []
        for record in self.workflow:
            if self.workflow[record] == [] and self.node_mapping[record].status == NodeStatus.WAITING:
                self.node_mapping[record].status = NodeStatus.EXECUTING
                self.node_mapping[record].start_time = time.time()
                node_list.append(self.node_mapping[record])
        return node_list

    def is_completed(self):
        return len(self.excuted_node) == self.node_count
    
    def print_info(self):
        print("excuted node:", self.excuted_node)


def create_dag_by_json(work_dir, node_dependency: list, node_list: list)->DAG:
    dag = DAG(work_dir)
    dag.init_workflow_by_json(node_dependency, node_list)
    return dag


if __name__ == "__main__":

    dependency_path = './data/node_dependency_0.json'
    list_path  = './data/node_list_0.json'

    json_path:str='./data/gaia/gaia_node_dependency.json'
    with open(dependency_path, "r", encoding="utf-8") as f:
        node_dep = json.load(f)

    json_path:str='./data/gaia/gaia_node_list.json'
    with open(list_path, "r", encoding="utf-8") as f:
        node_list = json.load(f)

    dependency = node_dep['./manual_001.json']
    lst = node_list['./manual_001.json']

    dag = create_dag_by_json("/root/nfs/LLMAgent/main_code",dependency,lst,"camel")

    print(dag.workflow)
