"""Async controller for CoreFlow.

The controller is a FastAPI application.  It receives queries, executes the
parsed DAG concurrently, and lets instances drive DAG advancement through
``/node_complete``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coreflow.dag.query import Query
from coreflow.instance_manager import InstanceManager
from coreflow.scheduler import AsyncAgentScheduler

logger = logging.getLogger("async_controller")


class QueryState:
    def __init__(self, query_id: int, query: Query) -> None:
        self.query_id = query_id
        self.query = query
        self.submit_time = time.time()
        self.completed = False
        self.error: Optional[str] = None


class AsyncController:
    def __init__(self) -> None:
        self.instance_manager = InstanceManager()
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self._schedulers: Dict[str, AsyncAgentScheduler] = {}
        self._queries: Dict[int, QueryState] = {}
        self._locks = {
            "schedulers": asyncio.Lock(),
            "queries": asyncio.Lock(),
        }
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._uvicorn_thread: Optional[threading.Thread] = None

    async def get_scheduler(self, agent_id: str) -> AsyncAgentScheduler:
        async with self._locks["schedulers"]:
            scheduler = self._schedulers.get(agent_id)
            if scheduler is None:
                scheduler = AsyncAgentScheduler(
                    agent_id=agent_id,
                    instance_manager=self.instance_manager,
                    http_client=self.http_client,
                )
                self._schedulers[agent_id] = scheduler
            return scheduler

    async def add_query(self, state: QueryState) -> None:
        async with self._locks["queries"]:
            self._queries[state.query_id] = state

    async def get_query(self, query_id: int) -> Optional[QueryState]:
        async with self._locks["queries"]:
            return self._queries.get(query_id)

    def load_allocation(self, path: str) -> None:
        """Pre-register instance stubs from an allocation file.

        Schedulers are created lazily by ``get_scheduler()``.  Real instances
        overwrite these stubs when they call ``/register``.
        """
        if not os.path.exists(path):
            logger.warning("Allocation file not found: %s", path)
            return

        with open(path) as handle:
            data = json.load(handle)

        allocation = data.get("allocation", {})
        gpus_per_instance = data.get("config", {}).get("gpus_per_instance", 1)

        for agent_id, agent_alloc in allocation.items():
            for instance_group in agent_alloc.get("instances", []):
                context_range = (
                    instance_group["context_lower"],
                    instance_group["context_upper"],
                )
                num_instances = instance_group.get("num_instances", 1)
                instance_gpus = instance_group.get(
                    "gpus_per_instance", gpus_per_instance
                )
                for index in range(num_instances):
                    self.instance_manager.register(
                        instance_id=(
                            f"{agent_id}-{context_range[0]}-"
                            f"{context_range[1]}-{index}"
                        ),
                        agent_id=agent_id,
                        host="127.0.0.1",
                        context_range=context_range,
                        num_gpus=instance_gpus,
                    )

    async def _schedule_nodes_async(
        self, state: QueryState, nodes: list
    ) -> None:
        tasks = [
            asyncio.create_task(self._dispatch_node_async(state, node))
            for node in nodes
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_node_async(self, state: QueryState, node: Any) -> None:
        node.query_id = state.query_id
        scheduler = await self.get_scheduler(node.agent_id)
        instance_id = await scheduler.route(
            query_id=node.query_id,
            invocation_id=node.invocation_id,
            input_tokens=node.input_tokens,
            num_input_tokens=node.num_input_tokens,
        )
        if instance_id is None:
            state.error = f"Routing failed for agent {node.agent_id}"
            return

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
        result = await scheduler.dispatch(instance_id, node_dict)
        if result is None or not result.get("success"):
            state.error = result.get("error") if result else "instance failed"

    async def execute_query_async(self, state: QueryState) -> None:
        try:
            nodes = state.query.get_entry_nodes()
            await self._schedule_nodes_async(state, nodes)
        except Exception as exc:
            state.error = str(exc)

    async def handle_node_complete(
        self,
        instance_id: Optional[str],
        query_id: int,
        invocation_id: int,
        repeat: int,
        agent_id: str,
        free_cache: bool,
        keep_cache_in_gpu: bool,
        input_token_ids: Optional[List[int]],
        output_token_ids: Optional[List[int]],
        num_input_tokens: Optional[int],
        num_output_tokens: Optional[int],
        success: bool,
        error: Optional[str],
    ) -> None:
        state = await self.get_query(query_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown query")

        if not success:
            state.error = error
            return

        if not free_cache:
            scheduler = await self.get_scheduler(agent_id)
            await scheduler.maybe_migrate_request(
                source_instance_id=instance_id,
                query_id=query_id,
                invocation_id=invocation_id,
                input_token_ids=input_token_ids or [],
                output_token_ids=output_token_ids or [],
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                keep_cache_in_gpu=keep_cache_in_gpu,
            )

        nodes = state.query.get_next_execute_agents(
            invocation_id=invocation_id,
            query_id=query_id,
            repeat=repeat,
        )
        if nodes:
            await self._schedule_nodes_async(state, nodes)

        if state.query.is_completed():
            state.completed = True
            logger.info(
                "Query %d completed in %.2f s",
                query_id,
                time.time() - state.submit_time,
            )

    async def close(self) -> None:
        await self.http_client.aclose()

    def wait(self) -> None:
        try:
            if self._uvicorn_thread is not None:
                self._uvicorn_thread.join()
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True


controller = AsyncController()
app = FastAPI()


class RegisterRequest(BaseModel):
    instance_id: str
    agent_id: str
    host: str = "127.0.0.1"
    port: Optional[int] = None
    context_range: Optional[tuple[int, int]] = None
    num_gpus: int = 1


class QueryRequest(BaseModel):
    query_id: Optional[int] = None
    node_dependency: dict = {}
    node_list: list = []
    work_dir: str = "/tmp"


class NodeCompleteRequest(BaseModel):
    instance_id: Optional[str] = None
    query_id: int
    invocation_id: int
    repeat: int = 0
    agent_id: str
    keep_cache_in_gpu: bool = False
    free_cache: bool = True
    success: bool = True
    error: Optional[str] = None
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None
    input_token_ids: Optional[List[int]] = None
    output_token_ids: Optional[List[int]] = None


@app.post("/register")
async def register(req: RegisterRequest):
    controller.instance_manager.register(
        instance_id=req.instance_id,
        agent_id=req.agent_id,
        host=req.host,
        port=req.port,
        context_range=req.context_range,
        num_gpus=req.num_gpus,
    )
    controller.instance_manager.set_ready(req.instance_id)
    await controller.get_scheduler(req.agent_id)
    return {"status": "registered", "instance_id": req.instance_id}


@app.post("/query")
async def create_query(req: QueryRequest):
    query_id = req.query_id or int(time.time() * 1000)
    query = Query(
        query_id=query_id,
        work_dir=req.work_dir,
        node_dependency=req.node_dependency,
        node_list=req.node_list,
        query_submit_time=time.time(),
    )
    state = QueryState(query_id, query)
    await controller.add_query(state)
    asyncio.create_task(controller.execute_query_async(state))
    return {"status": "accepted", "query_id": query_id}


@app.post("/node_complete")
async def node_complete(req: NodeCompleteRequest):
    await controller.handle_node_complete(
        instance_id=req.instance_id,
        query_id=req.query_id,
        invocation_id=req.invocation_id,
        repeat=req.repeat,
        agent_id=req.agent_id,
        free_cache=req.free_cache,
        keep_cache_in_gpu=req.keep_cache_in_gpu,
        input_token_ids=req.input_token_ids,
        output_token_ids=req.output_token_ids,
        num_input_tokens=req.num_input_tokens,
        num_output_tokens=req.num_output_tokens,
        success=req.success,
        error=req.error,
    )
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_controller(
    host: str = "0.0.0.0",
    port: int = 5000,
    allocation_path: str = "allocation.json",
) -> AsyncController:
    """Create/start the async controller and return a handle for wait()/stop()."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if allocation_path:
        controller.load_allocation(allocation_path)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    controller._uvicorn_server = server
    controller._uvicorn_thread = thread
    return controller


if __name__ == "__main__":
    ctrl = run_controller()
    ctrl.wait()
