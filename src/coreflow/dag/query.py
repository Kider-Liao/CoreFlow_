from collections import defaultdict
from coreflow.dag.dag import DAG, Node, create_dag_by_json
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import json

class Query:
    def __init__(
        self,
        query_id: int,
        work_dir: str,
        node_dependency:dict,
        node_list:list ,
        query_submit_time:float = 0.0,
    ):
        self.query_id = query_id
        self.node_dependency = node_dependency
        self.node_list = node_list
        self.query_submit_time = query_submit_time
        self.work_dir = work_dir
        self.dag = create_dag_by_json(work_dir=work_dir, node_dependency=node_dependency, node_list=node_list)
        self.agent_cumulative_time = 0

    def get_entry_nodes(
        self
    ) -> List[Node]:
        return self.dag.get_entry_nodes()

    def get_entry(
        self
    ):  
        return self.dag.entry
    
    def get_next_execute_agents(
        self,
        invocation_id:int,
        query_id:int,
        repeat:int
    )->List[Node]:
        node_key = (invocation_id, repeat)
        if node_key in self.dag.node_mapping:
            self.agent_cumulative_time += time.time() - self.dag.node_mapping[node_key].start_time
        return self.dag.update_result(invocation_id, query_id, repeat)
    
    def is_completed(
        self
    ):
        return self.dag.is_completed()
    
    def get_latency(self):
        assert self.is_completed()
        assert self.dag.end_to_end_latency is not None
        return self.dag.end_to_end_latency

    def print_info(self):
        return self.dag.print_info()

    def __str__(self):
        return f"Request<query_id={self.query_id}>"
        
    def __repr__(self):
        return f"Request<query_id={self.query_id}>"