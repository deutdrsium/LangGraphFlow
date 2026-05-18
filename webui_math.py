from contextlib import redirect_stderr, redirect_stdout
from collections import deque
import io
import json
import os
import re
import sys
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel
import uvicorn

from rate_limiter import instance_limiter, pro_limiter

load_dotenv()

MATH_DEBUG = "--debug" in sys.argv or "-debug" in sys.argv

MAX_INSTANCES = min(int(os.getenv("MAX_INSTANCES", "8")), 8)
if MAX_INSTANCES < 1:
    MAX_INSTANCES = 1


MATH_MODEL_TIMEOUT = float(os.getenv("MATH_MODEL_TIMEOUT", os.getenv("MODEL_TIMEOUT", "300")))
if MATH_MODEL_TIMEOUT < 60:
    MATH_MODEL_TIMEOUT = 60.0

import main
from main import (
    GraphState,
    code_executor_node,
    graph_app as debug_graph_app,
    judge_node,
    route_after_analyze,
    route_after_judge,
    type_classifier_node,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

sessions = {}
task_queue = deque()


class TaskData(BaseModel):
    task_id: str
    question_content: str
    answer: str


class CancelData(BaseModel):
    task_id: str


class RunData(BaseModel):
    question: str
    truth: str
    task_id: str = None


class HumanDecision(BaseModel):
    decision: str


def analyze_and_solve_non_stream_node(state: GraphState, config=None):
    question = state.get("question_context", "")
    problem_type = state.get("problem_type", "代数")

    prompt_env_map = {
        "几何": "ANALYZE_AND_SOLVE_PROMPT_几何",
        "代数": "ANALYZE_AND_SOLVE_PROMPT_代数",
        "概率": "ANALYZE_AND_SOLVE_PROMPT_概率",
        "数论": "ANALYZE_AND_SOLVE_PROMPT_数论",
    }
    env_key = prompt_env_map.get(problem_type, "ANALYZE_AND_SOLVE_PROMPT_代数")
    system_prompt = os.getenv(env_key, "You are a helpful math assistant.")

    client = OpenAI()
    attempt = 0
    error_codes = []

    while True:
        try:
            if pro_limiter:
                pro_limiter.acquire()
            response = client.chat.completions.create(
                model=os.getenv("MODEL_PRO", "gemini-3.1-pro-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析并求解以下题目：\n\n{question}"},
                ],
                temperature=1.0,
                stream=False,
                timeout=MATH_MODEL_TIMEOUT,
            )

            final_content = response.choices[0].message.content or ""

            if "[TRAP_DETECTED]" in final_content:
                trap_reason = final_content.split("[TRAP_DETECTED]", 1)[1].strip()
                if not trap_reason:
                    trap_reason = "模型检测到逻辑陷阱但未给出原因"
                return {
                    "trap_analysis": True,
                    "trap_reason": trap_reason,
                    "generated_code": "pass",
                }

            match = re.search(r"```python\n(.*?)\n```", final_content, re.DOTALL)
            generated_code = match.group(1).strip() if match else final_content
            return {
                "trap_analysis": False,
                "trap_reason": "pass",
                "generated_code": generated_code,
            }

        except Exception as e:
            attempt += 1
            is_429 = (hasattr(e, "status_code") and e.status_code == 429) or "429" in str(e)
            error_codes.append(429 if is_429 else 0)

            if attempt >= 3:
                if all(code == 429 for code in error_codes[-3:]):
                    import time

                    time.sleep(15)
                    attempt = 0
                    error_codes.clear()
                    continue
                raise Exception(f"analyze_and_solve 连续失败3次（非全为429限流），阻塞: {e}") from e


def _quiet_node(fn):
    def wrapped(*args, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)

    return wrapped


def _build_math_graph(debug: bool):
    if debug:
        return debug_graph_app

    workflow = StateGraph(GraphState)
    workflow.add_node("type_classifier", _quiet_node(type_classifier_node))
    workflow.add_node("analyze_and_solve", _quiet_node(analyze_and_solve_non_stream_node))
    workflow.add_node("code_executor", _quiet_node(code_executor_node))
    workflow.add_node("judge", _quiet_node(judge_node))
    workflow.add_node("human_review", _quiet_node(lambda state: {}))
    workflow.add_edge(START, "type_classifier")
    workflow.add_edge("type_classifier", "analyze_and_solve")
    workflow.add_conditional_edges(
        "analyze_and_solve",
        _quiet_node(route_after_analyze),
        {"end": END, "code_executor": "code_executor"},
    )
    workflow.add_edge("code_executor", "judge")
    workflow.add_conditional_edges(
        "judge",
        _quiet_node(route_after_judge),
        {"end": END, "human_review": "human_review"},
    )
    workflow.add_edge("human_review", END)
    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])


graph_app = _build_math_graph(MATH_DEBUG)


@app.get("/")
async def get_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/config")
async def get_config():
    return {"max_instances": MAX_INSTANCES}


@app.post("/api/task_data")
async def receive_task_data(data: TaskData):
    if instance_limiter:
        if MATH_DEBUG:
            print(f"[Rate Limit] Task {data.task_id} waiting for INSTANCE slot...", flush=True)
        await instance_limiter.async_acquire()
    if MATH_DEBUG:
        print("\n--- New Data Received! ---")
        print(f"Task ID: {data.task_id}")
        print(f"Question Length: {len(data.question_content)} chars")
        print("--------------------------\n")
    task_queue.append(data)
    return {"status": "success", "message": "Data queued successfully"}


@app.post("/api/cancel_task")
async def cancel_task(data: CancelData):
    global task_queue
    original_len = len(task_queue)
    task_queue = deque([task for task in task_queue if task.task_id != data.task_id])
    found = len(task_queue) < original_len

    for thread_id, session in sessions.items():
        if session.get("state", {}).get("question_id") == data.task_id:
            session["status"] = "cancelled"
            found = True

    if MATH_DEBUG and not found:
        print(f"Warning: No valid session running for Task {data.task_id}")

    return {"status": "success", "message": "Task cancelled"}


@app.get("/api/get_task")
async def get_task():
    if task_queue:
        task = task_queue.popleft()
        return {"has_task": True, "task": task.model_dump()}
    return {"has_task": False}


@app.get("/api/active_sessions")
async def get_active_sessions():
    active = []
    for tid, session in sessions.items():
        if session.get("status") not in ["cancelled"]:
            active.append(
                {
                    "thread_id": tid,
                    "task_id": session.get("state", {}).get("question_id", ""),
                    "question": session.get("state", {}).get("question_context", ""),
                    "truth": session.get("state", {}).get("ground_truth", ""),
                    "status": session.get("status"),
                    "nodes": session.get("nodes", {}),
                }
            )
    return {"sessions": active[:MAX_INSTANCES]}


@app.get("/api/task_result/{task_id}")
async def get_task_result(task_id: str):
    for session in sessions.values():
        if session.get("state", {}).get("question_id") == task_id:
            if session.get("status") == "finished":
                return {"status": "finished", "data": session.get("state")}
            if session.get("status") == "blocked":
                state_data = session.get("state").copy()
                state_data["final_decision"] = "HITL"
                state_data["hitl_reason"] = (
                    f"系统置信度低 ({state_data.get('confidence_score', 0)} < 75)，需要人工介入审核"
                )
                state_data["code_execution_result"] = state_data.get("execution_output", "")
                return {"status": "finished", "data": state_data}
            if session.get("status") == "running":
                return {"status": "running"}
    return {"status": "not_found"}


def _mark_next_nodes(thread_id: str, config: dict):
    current_state = graph_app.get_state(config)
    for next_node in current_state.next:
        if next_node == "human_review":
            continue
        if next_node not in sessions[thread_id]["nodes"]:
            sessions[thread_id]["nodes"][next_node] = {"status": "executing"}
        elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
            sessions[thread_id]["nodes"][next_node]["status"] = "executing"


def _finish_or_block(thread_id: str, config: dict):
    state_info = graph_app.get_state(config)
    sessions[thread_id]["state"].update(state_info.values or {})
    needs_hitl = state_info.next and "human_review" in state_info.next
    if needs_hitl:
        sessions[thread_id]["status"] = "blocked"
        sessions[thread_id]["nodes"]["human_review"] = {
            "status": "blocked",
            "data": {"message": "Waiting for manual review..."},
        }
    else:
        sessions[thread_id]["status"] = "finished"
        sessions[thread_id]["nodes"]["__end__"] = {
            "status": "success",
            "data": sessions[thread_id]["state"],
        }


def _run_graph_debug(thread_id: str, config: dict, initial_state: dict):
    sessions[thread_id]["nodes"]["type_classifier"] = {"status": "executing"}
    for update in graph_app.stream(initial_state, config=config, stream_mode="updates"):
        if sessions[thread_id].get("status") == "cancelled":
            return

        for node_name, node_update in update.items():
            if node_update is None:
                node_update = {}
            sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update}
            sessions[thread_id]["state"].update(node_update)
        _mark_next_nodes(thread_id, config)

    _finish_or_block(thread_id, config)


def _run_graph_quiet(thread_id: str, config: dict, initial_state: dict):
    sessions[thread_id]["nodes"]["type_classifier"] = {"status": "executing"}
    graph_app.invoke(initial_state, config=config)

    if sessions[thread_id].get("status") == "cancelled":
        return

    state_values = graph_app.get_state(config).values or {}
    sessions[thread_id]["state"].update(state_values)

    for node_name in ("type_classifier", "analyze_and_solve", "code_executor", "judge"):
        data = {}
        if node_name == "type_classifier":
            data = {
                key: sessions[thread_id]["state"].get(key)
                for key in ("problem_type", "hierarchy", "difficulty")
                if key in sessions[thread_id]["state"]
            }
        elif node_name == "analyze_and_solve":
            data = {
                key: sessions[thread_id]["state"].get(key)
                for key in ("trap_analysis", "trap_reason", "generated_code")
                if key in sessions[thread_id]["state"]
            }
        elif node_name == "code_executor" and "execution_output" in sessions[thread_id]["state"]:
            data = {"execution_output": sessions[thread_id]["state"].get("execution_output")}
        elif node_name == "judge" and "confidence_score" in sessions[thread_id]["state"]:
            data = {
                key: sessions[thread_id]["state"].get(key)
                for key in ("confidence_score", "final_decision", "verified_ans")
                if key in sessions[thread_id]["state"]
            }

        if data:
            sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": data}

    _finish_or_block(thread_id, config)


@app.post("/api/run")
async def start_run(data: RunData):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question_id": data.task_id if data.task_id else f"q_{thread_id[:4]}",
        "question_context": data.question,
        "ground_truth": data.truth,
    }

    sessions[thread_id] = {
        "status": "running",
        "nodes": {"__start__": {"status": "success", "data": initial_state}},
        "state": initial_state,
    }

    def run_graph():
        try:
            if MATH_DEBUG:
                print(f"\n[WebUI-Math] Received task: {data.question[:20]}...", flush=True)
                _run_graph_debug(thread_id, config, initial_state)
            else:
                _run_graph_quiet(thread_id, config, initial_state)
        except Exception as e:
            if MATH_DEBUG:
                import traceback

                traceback.print_exc()
            sessions[thread_id]["status"] = "error"
            sessions[thread_id]["error"] = str(e)
            for node_info in sessions[thread_id]["nodes"].values():
                if isinstance(node_info, dict) and node_info.get("status") == "executing":
                    node_info["status"] = "error"
                    node_info["data"] = {"error": str(e)}

    threading.Thread(target=run_graph, daemon=True).start()
    return {"thread_id": thread_id}


@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str):
    if thread_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    if MATH_DEBUG and thread_id in main.streaming_store:
        if "analyze_and_solve" in sessions[thread_id]["nodes"]:
            node_info = sessions[thread_id]["nodes"]["analyze_and_solve"]
            if node_info.get("status") == "executing" or "generated_code" not in node_info.get("data", {}):
                node_info["data"] = {"streaming_content": main.streaming_store[thread_id]}

    return sessions[thread_id]


@app.post("/api/resume/{thread_id}")
async def resume_run(thread_id: str, data: HumanDecision):
    if thread_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    session = sessions[thread_id]
    if session["status"] != "blocked":
        return JSONResponse(status_code=400, content={"error": "Session is not blocked"})

    config = {"configurable": {"thread_id": thread_id}}
    graph_app.update_state(config, {"final_decision": data.decision})

    session["status"] = "running"
    session["nodes"]["human_review"] = {"status": "success", "data": {"final_decision": data.decision}}

    def resume_graph():
        try:
            if MATH_DEBUG:
                _mark_next_nodes(thread_id, config)
                for update in graph_app.stream(None, config=config, stream_mode="updates"):
                    if sessions[thread_id].get("status") == "cancelled":
                        return
                    for node_name, node_update in update.items():
                        if node_update is None:
                            node_update = {}
                        sessions[thread_id]["nodes"][node_name] = {
                            "status": "success",
                            "data": node_update,
                        }
                        sessions[thread_id]["state"].update(node_update)
                    _mark_next_nodes(thread_id, config)
            else:
                graph_app.invoke(None, config=config)

            sessions[thread_id]["state"].update(graph_app.get_state(config).values or {})
            sessions[thread_id]["status"] = "finished"
            sessions[thread_id]["nodes"]["__end__"] = {
                "status": "success",
                "data": sessions[thread_id]["state"],
            }
        except Exception as e:
            if MATH_DEBUG:
                import traceback

                traceback.print_exc()
            sessions[thread_id]["status"] = "error"
            sessions[thread_id]["error"] = str(e)

    threading.Thread(target=resume_graph, daemon=True).start()
    return {"status": "success"}


if __name__ == "__main__":
    if MATH_DEBUG:
        print("[WebUI-Math] DEBUG mode enabled; using original streaming graph.", flush=True)
    sys.argv = [a for a in sys.argv if a not in ("--debug", "-debug")]
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=MATH_DEBUG)
