"""Controller: HTTP server that orchestrates multi-agent workflow execution.

Key endpoints:
  POST /register      – Instance registers itself after startup.
  POST /query         – Submit a new Query for execution.
  POST /node_complete – Instance notifies controller a node finished.

The controller manages:
  - InstanceManager   – Instance registry and failover.
  - AgentScheduler×N  – Per-agent routing and load balancing.
  - Query×M           – In-flight DAG execution state.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from coreflow.instance_manager import InstanceManager
from coreflow.scheduler import AgentScheduler

logger = logging.getLogger("controller")


# ---------------------------------------------------------------------------
# Query execution state (in-memory per-query DAG tracker)
# ---------------------------------------------------------------------------
class QueryState:
    """Tracks the execution state of a single Query's DAG."""

    def __init__(self, query_id: int, query_obj) -> None:
        self.query_id = query_id
        self.query_obj = query_obj  # DAG.Query instance
        self.submit_time = time.time()
        self.completed = False
        self.pending_nodes: Dict[tuple, Any] = {}  # (invocation_id, repeat) -> Node
        self.error: Optional[str] = None


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class ControllerHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the controller."""

    # Class-level refs set by Controller.__init__
    controller: "Controller" = None  # type: ignore[assignment]

    def log_message(self, format, *args):
        logger.info("%s - %s", self.client_address[0], format % args)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400) -> None:
        self._send_json({"error": msg}, status)

    # ------------------------------------------------------------------
    # POST /register
    # ------------------------------------------------------------------
    def do_POST_register(self) -> None:
        """Instance registers itself after startup."""
        try:
            body = self._read_body()
        except Exception:
            self._send_error("Invalid JSON body")
            return

        required = ["instance_id", "agent_id"]
        for field in required:
            if field not in body:
                self._send_error(f"Missing required field: {field}")
                return

        info = self.controller.instance_manager.register(
            instance_id=body["instance_id"],
            agent_id=body["agent_id"],
            host=body.get("host", "127.0.0.1"),
            port=body.get("port"),
            context_range=body.get("context_range"),
            num_gpus=body.get("num_gpus", 1),
        )
        # Mark ready
        self.controller.instance_manager.set_ready(info.instance_id)

        # Ensure scheduler exists for this agent
        self.controller.get_scheduler(info.agent_id)

        logger.info(
            "Instance registered: %s agent=%s endpoint=%s ranges=%s",
            info.instance_id, info.agent_id, info.endpoint, info.context_range,
        )
        self._send_json({
            "status": "registered",
            "instance_id": info.instance_id,
            "endpoint": info.endpoint,
        })

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------
    def do_POST_query(self) -> None:
        """Accept a new query for execution."""
        try:
            body = self._read_body()
        except Exception:
            self._send_error("Invalid JSON body")
            return

        # Build Query from DAG module
        try:
            from coreflow.dag.query import Query
        except ImportError:
            self._send_error("DAG module not available", 500)
            return

        query_id = body.get("query_id") or int(time.time() * 1000)
        node_dependency = body.get("node_dependency", {})
        node_list = body.get("node_list", [])
        work_dir = body.get("work_dir", "/tmp")

        try:
            query = Query(
                query_id=query_id,
                work_dir=work_dir,
                node_dependency=node_dependency,
                node_list=node_list,
                query_submit_time=time.time(),
            )
        except Exception as exc:
            self._send_error(f"Failed to create Query: {exc}")
            return

        state = QueryState(query_id, query)
        with self.controller._query_lock:
            self.controller._queries[query_id] = state

        logger.info("Query %d accepted", query_id)
        self._send_json({"status": "accepted", "query_id": query_id})

        # Kick off execution in background thread
        threading.Thread(
            target=self.controller._execute_query,
            args=(state,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # POST /node_complete
    # ------------------------------------------------------------------
    def do_POST_node_complete(self) -> None:
        """Instance notifies controller that a node finished executing."""
        try:
            body = self._read_body()
        except Exception:
            self._send_error("Invalid JSON body")
            return

        query_id = body.get("query_id")
        invocation_id = body.get("invocation_id")
        repeat = body.get("repeat", 0)
        agent_id = body.get("agent_id")
        keep_cache_in_gpu = body.get("keep_cache_in_gpu", False)
        free_cache = body.get("free_cache", True)
        num_input_tokens = body.get("num_input_tokens", 0)
        num_output_tokens = body.get("num_output_tokens", 0)
        success = body.get("success", True)
        error = body.get("error")

        if query_id is None or invocation_id is None or agent_id is None:
            self._send_error("Missing required fields")
            return

        # Get query state
        state = self.controller._queries.get(query_id)
        if state is None:
            self._send_error(f"Unknown query: {query_id}")
            return

        if not success:
            logger.error(
                "Node failed: qid=%d iid=%d:%d err=%s",
                query_id, invocation_id, repeat, error,
            )
            state.error = error
            self._send_json({"status": "error_recorded"})
            return

        # Handle KV cache migration if needed
        scheduler = self.controller.get_scheduler(agent_id)
        if scheduler is None:
            self._send_error(f"No scheduler for agent: {agent_id}")
            return

        if free_cache:
            scheduler.release(query_id, invocation_id)
        elif keep_cache_in_gpu:
            migration = scheduler.check_and_migrate(
                query_id=query_id,
                invocation_id=invocation_id,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
            )
            if migration:
                logger.info(
                    "KV migration initiated: qid=%d iid=%d %s -> %s",
                    query_id, invocation_id,
                    migration["src_instance_id"],
                    migration["dst_instance_id"],
                )
        # Advance DAG
        self.controller._advance_dag(state, invocation_id, repeat)

        self._send_json({"status": "ok"})

    # ------------------------------------------------------------------
    # GET /status
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/status":
            stats = {
                "queries": len(self.controller._queries),
                "instances": len(self.controller.instance_manager.all_instance_ids()),
                "schedulers": {
                    aid: s.stats()
                    for aid, s in self.controller._schedulers.items()
                },
            }
            self._send_json(stats)
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_error("Not found", 404)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        routes = {
            "/register": self.do_POST_register,
            "/query": self.do_POST_query,
            "/node_complete": self.do_POST_node_complete,
        }
        handler = routes.get(self.path)
        if handler:
            handler()
        else:
            self._send_error(f"Unknown endpoint: {self.path}", 404)


# ---------------------------------------------------------------------------
# Controller (main orchestrator)
# ---------------------------------------------------------------------------
class Controller:
    """Top-level controller for workflow execution.

    Responsibilities:
      - Run HTTP server (ControllerHandler).
      - Manage InstanceManager and AgentScheduler instances.
      - Coordinate DAG execution: route nodes → schedulers → instances.
      - Handle KV cache migration decisions.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port

        self.instance_manager = InstanceManager()

        # agent_id -> AgentScheduler
        self._schedulers: Dict[str, AgentScheduler] = {}
        self._scheduler_lock = threading.Lock()

        # query_id -> QueryState
        self._queries: Dict[int, QueryState] = {}
        self._query_lock = threading.Lock()

        self._http_server: Optional[ThreadingHTTPServer] = None
        self._running = threading.Event()

    # ------------------------------------------------------------------
    # Allocation loading
    # ------------------------------------------------------------------
    def load_allocation(self, path: str = "allocation.json") -> None:
        """Load allocation and pre-build schedulers + instance stubs.

        Reads the allocation JSON produced by the resource allocator.
        For each agent's instance group, creates placeholder InstanceInfo
        entries so the scheduler routing tables are ready when instances
        register themselves.

        Args:
            path: Path to the allocation JSON file.
        """
        alloc_path = Path(path)
        if not alloc_path.exists():
            logger.warning("Allocation file not found: %s", alloc_path)
            return

        with open(alloc_path) as f:
            data = json.load(f)

        allocation = data.get("allocation", {})
        gpus_per_instance = data.get("config", {}).get("gpus_per_instance", 1)

        for agent_id, agent_alloc in allocation.items():
            self.get_scheduler(agent_id)

            for ig in agent_alloc.get("instances", []):
                context_range = (ig["context_lower"], ig["context_upper"])
                num_instances = ig.get("num_instances", 1)
                instance_gpus = ig.get("gpus_per_instance", gpus_per_instance)

                for i in range(num_instances):
                    inst_id = f"{agent_id}-{context_range[0]}-{context_range[1]}-{i}"
                    self.instance_manager.register(
                        instance_id=inst_id,
                        agent_id=agent_id,
                        host="127.0.0.1",
                        context_range=context_range,
                        num_gpus=instance_gpus,
                    )
                logger.info(
                    "Allocation: agent=%s range=[%d,%d) instances=%d gpus=%d",
                    agent_id, *context_range, num_instances, instance_gpus,
                )

        logger.info("Allocation loaded from %s (%d agents)", path, len(allocation))

    # ------------------------------------------------------------------
    # Scheduler access
    # ------------------------------------------------------------------
    def get_scheduler(self, agent_id: str) -> AgentScheduler:
        """Get or create the scheduler for an agent."""
        with self._scheduler_lock:
            if agent_id not in self._schedulers:
                self._schedulers[agent_id] = AgentScheduler(
                    agent_id=agent_id,
                    instance_manager=self.instance_manager,
                )
            return self._schedulers[agent_id]

    # ------------------------------------------------------------------
    # HTTP server lifecycle
    # ------------------------------------------------------------------
    def start(self, allocation_path: str = "allocation.json") -> None:
        """Start the HTTP server in a background thread."""
        ControllerHandler.controller = self

        self.load_allocation(allocation_path)

        self._http_server = ThreadingHTTPServer(
            (self.host, self.port), ControllerHandler
        )
        self._running.set()

        logger.info("Controller listening on %s:%d", self.host, self.port)

        thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        self._running.clear()
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
        logger.info("Controller stopped")

    def wait(self) -> None:
        """Block until the server is stopped."""
        try:
            while self._running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    # ------------------------------------------------------------------
    # DAG execution
    # ------------------------------------------------------------------
    def _execute_query(self, state: QueryState) -> None:
        """Background worker that drives DAG execution for a single query."""
        try:
            # Get initial batch of entry nodes (no dependencies)
            nodes = state.query_obj.get_entry_nodes()
            # Set query_id (invocation_id already set by DAG parser)
            for node in nodes:
                node.query_id = state.query_id

            self._dispatch_nodes(state, nodes)
        except Exception as exc:
            logger.error(
                "Query %d execution error: %s\n%s",
                state.query_id, exc, traceback.format_exc(),
            )
            state.error = str(exc)

    def _advance_dag(
        self, state: QueryState, invocation_id: int, repeat: int
    ) -> None:
        """After a node completes, advance the DAG and dispatch next nodes."""
        if state.completed or state.error:
            return

        try:
            nodes = state.query_obj.get_next_execute_agents(
                invocation_id=invocation_id,
                query_id=state.query_id,
                repeat=repeat,
            )
        except Exception as exc:
            logger.error("DAG advance error: %s", exc)
            state.error = str(exc)
            return

        if not nodes:
            # Check if query is fully completed
            if state.query_obj.is_completed():
                state.completed = True
                latency = time.time() - state.submit_time
                logger.info(
                    "Query %d completed in %.2f s",
                    state.query_id, latency,
                )
            return

        # Assign query_id; invocation_id already set by DAG
        for node in nodes:
            node.query_id = state.query_id

        self._dispatch_nodes(state, nodes)

    def _dispatch_nodes(
        self, state: QueryState, nodes: list
    ) -> None:
        """Route and dispatch a batch of nodes to vLLM instances."""
        for node in nodes:
            scheduler = self.get_scheduler(node.agent_id)

            # Route this invocation
            instance_id = scheduler.route(
                query_id=node.query_id,
                invocation_id=node.invocation_id,
                num_input_tokens=node.num_input_tokens,
            )
            if instance_id is None:
                logger.error(
                    "Failed to route qid=%d iid=%d agent=%s",
                    node.query_id, node.invocation_id, node.agent_id,
                )
                state.error = f"Routing failed for agent {node.agent_id}"
                return

            # Build dispatch payload
            node_dict = {
                "query_id": node.query_id,
                "invocation_id": node.invocation_id,
                "agent_id": node.agent_id,
                "input_tokens": node.input_tokens,
                "output_tokens": node.output_tokens,
                "num_input_tokens": node.num_input_tokens,
                "num_output_tokens": node.num_output_tokens,
                "keep_cache_in_gpu": node.keep_cache_in_gpu,
                "free_cache": node.free_cache,
                "repeat": node.repeat,
            }

            # Dispatch asynchronously (fire-and-forget per node)
            threading.Thread(
                target=self._dispatch_one,
                args=(scheduler, instance_id, node_dict, state, node),
                daemon=True,
            ).start()

    def _dispatch_one(
        self,
        scheduler: AgentScheduler,
        instance_id: str,
        node_dict: dict,
        state: QueryState,
        node,
    ) -> None:
        """Dispatch a single node and handle the response."""
        try:
            result = scheduler.dispatch(instance_id, node_dict)
            if result is None or not result.get("success"):
                # Failover: try alternative instance in same context range
                slot = self.instance_manager.find_context_slot(
                    node.agent_id, node.num_input_tokens,
                )
                if slot is None:
                    state.error = (
                        f"No context slot for agent={node.agent_id} "
                        f"input={node.num_input_tokens}"
                    )
                    return
                alt = self.instance_manager.find_alternative(
                    node.agent_id, slot, instance_id,
                )
                if alt:
                    logger.warning(
                        "Failover: %s -> %s for qid=%d",
                        instance_id, alt.instance_id, node.query_id,
                    )
                    result2 = scheduler.dispatch(alt.instance_id, node_dict)
                    if result2 is None or not result2.get("success"):
                        state.error = (
                            f"Both primary ({instance_id}) and "
                            f"alternative ({alt.instance_id}) failed"
                        )
                else:
                    state.error = f"Instance {instance_id} failed, no alternative"
        except Exception as exc:
            logger.error("Dispatch error: %s", exc)
            state.error = str(exc)

    # ------------------------------------------------------------------
    # Instance failover
    # ------------------------------------------------------------------
    def handle_instance_failure(self, instance_id: str) -> None:
        """Called when an instance is detected as failed.

        Marks it unavailable, restarts it (externally), and transfers
        active invocations to alternatives in the same context range.
        """
        info = self.instance_manager.mark_unavailable(instance_id)
        if info is None:
            return

        logger.warning("Instance %s marked unavailable, initiating failover", instance_id)

        # Try to restart (external process management — placeholder)
        # In a real system this would launch a new vLLM process.
        logger.info("Restarting instance %s (external)", instance_id)

        # Transfer active invocations to alternatives
        scheduler = self._schedulers.get(info.agent_id)
        if scheduler is None:
            return

        # Collect affected (qid, iid) pairs
        affected: List[tuple] = []
        with scheduler._routing_lock:
            for (qid, iid), iid2 in list(scheduler._routing.items()):
                if iid2 == instance_id:
                    affected.append((qid, iid))

        for qid, iid in affected:
            # invocation_id identifies the session; scan repeats for token count
            inv_id = iid
            state = self._queries.get(qid)
            num_tokens = 0
            if state:
                for repeat in range(100):  # scan limited range for any repeat
                    key = (inv_id, repeat)
                    if key in state.query_obj.dag.node_mapping:
                        node = state.query_obj.dag.node_mapping[key]
                        num_tokens = len(node.input_tokens)
                        break

            slot = self.instance_manager.find_context_slot(
                info.agent_id, num_tokens
            )
            if slot is None:
                continue
            alt = self.instance_manager.find_alternative(
                info.agent_id, slot, instance_id,
            )
            if alt:
                with scheduler._routing_lock:
                    scheduler._routing[(qid, iid)] = alt.instance_id
                scheduler._add_load(alt.instance_id, qid, iid)
                scheduler._remove_load(instance_id, qid, iid)
                logger.info(
                    "Failover qid=%d iid=%d: %s -> %s",
                    qid, iid, instance_id, alt.instance_id,
                )

        # Remove dead instance
        self.instance_manager.remove(instance_id)


# ---------------------------------------------------------------------------
# Convenience: run as standalone
# ---------------------------------------------------------------------------
def run_controller(
    host: str = "0.0.0.0",
    port: int = 5000,
    allocation_path: str = "allocation.json",
) -> Controller:
    """Create and start a controller, returning it for programmatic use."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctrl = Controller(host=host, port=port)
    ctrl.start(allocation_path=allocation_path)
    return ctrl


if __name__ == "__main__":
    ctrl = run_controller()
    ctrl.wait()
