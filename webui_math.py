from contextlib import redirect_stderr, redirect_stdout
from collections import deque
import asyncio
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import docker
import requests
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

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MATH_DEBUG = "--debug" in sys.argv or "-debug" in sys.argv

MAX_INSTANCES = min(int(os.getenv("MAX_INSTANCES", "8")), 8)
if MAX_INSTANCES < 1:
    MAX_INSTANCES = 1


MATH_MODEL_TIMEOUT = float(os.getenv("MATH_MODEL_TIMEOUT", os.getenv("MODEL_TIMEOUT", "300")))
if MATH_MODEL_TIMEOUT < 60:
    MATH_MODEL_TIMEOUT = 60.0

MODEL_LITE_TIMEOUT = float(os.getenv("MODEL_LITE_TIMEOUT", "60"))
if MODEL_LITE_TIMEOUT < 10:
    MODEL_LITE_TIMEOUT = 10.0

MATH_CONDA_BASE_IMAGE = os.getenv("MATH_CONDA_BASE_IMAGE", "anaconda/miniconda:latest")
MATH_CONDA_ENV_NAME = os.getenv("MATH_CONDA_ENV_NAME", "math_env")
MATH_CONDA_ENVS_VOLUME = os.getenv("MATH_CONDA_ENVS_VOLUME", "langgraph_math_conda_envs")
MATH_CONDA_PIP_CACHE_VOLUME = os.getenv("MATH_CONDA_PIP_CACHE_VOLUME", "langgraph_math_pip_cache")
MATH_CONDA_INITIAL_PACKAGES = os.getenv(
    "MATH_CONDA_INITIAL_PACKAGES",
    "numpy sympy z3-solver networkx matplotlib scipy",
).split()
MATH_CODE_TIMEOUT = int(os.getenv("MATH_CODE_TIMEOUT", "60"))
MATH_CONDA_INSTALL_TIMEOUT = int(os.getenv("MATH_CONDA_INSTALL_TIMEOUT", "600"))
WOLFRAM_BRIDGE_ENABLED = os.getenv("WOLFRAM_BRIDGE_ENABLED", "true").lower() == "true"
WOLFRAM_BRIDGE_HOST = os.getenv("WOLFRAM_BRIDGE_HOST", "127.0.0.1")
WOLFRAM_BRIDGE_PORT = int(os.getenv("WOLFRAM_BRIDGE_PORT", "8765"))
WOLFRAM_BRIDGE_CONNECT_HOST = os.getenv("WOLFRAM_BRIDGE_CONNECT_HOST", "host.docker.internal")
WOLFRAM_BRIDGE_TOKEN = os.getenv("WOLFRAM_BRIDGE_TOKEN", uuid.uuid4().hex)
WOLFRAM_SCRIPT_PATH = os.getenv("WOLFRAM_SCRIPT_PATH", "")
_conda_env_lock = threading.Lock()
_wolfram_bridge_server = None
_wolfram_bridge_lock = threading.Lock()
_wolfram_process_lock = threading.Lock()
MATH_BUILTIN_HELPER_MODULES = {"wolfram_eval"}

import main
from main import (
    ClassificationResult,
    GraphState,
    analyze_and_solve_node,
    extract_python_code,
    judge_node,
    code_extraction_max_retries,
    code_retry_user_prompt,
    reasoning_effort_kwargs,
    route_after_analyze,
    route_after_judge,
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


def type_classifier_lite_node(state: GraphState):
    question = state.get("question_context", "")
    prompt = os.getenv("TYPE_CLASSIFIER_PROMPT", "You are an expert math problem classifier...")
    model_name = os.getenv("MODEL_LITE", "claude-3-5-haiku-latest")
    support_structured = os.getenv("SUPPORT_STRUCTURED_OUTPUT", "True").lower() == "true"
    client = OpenAI()

    try:
        if support_structured:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Please classify the following question:\n\n{question}"},
                ],
                response_format=ClassificationResult,
                temperature=1.0,
                timeout=MODEL_LITE_TIMEOUT,
                **reasoning_effort_kwargs("LITE"),
            )
            result = response.choices[0].message.parsed
            return {
                "problem_type": result.problem_type,
                "hierarchy": result.hierarchy,
                "difficulty": result.difficulty,
            }

        prompt_with_instructions = (
            prompt
            + '\n\nPlease return ONLY a JSON object string exactly matching this schema: '
            + '{"problem_type": "几何|代数|概率|数论", "hierarchy": "初中|高中|本科|硕士及以上", '
            + '"difficulty": "基础|进阶|竞赛"}. Do not use code blocks.'
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt_with_instructions},
                {"role": "user", "content": f"Please classify the following question:\n\n{question}"},
            ],
            temperature=1.0,
            timeout=MODEL_LITE_TIMEOUT,
            **reasoning_effort_kwargs("LITE"),
        )
        content = (response.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(content)
        return {
            "problem_type": result_dict.get("problem_type", "代数"),
            "hierarchy": result_dict.get("hierarchy", "高中"),
            "difficulty": result_dict.get("difficulty", "基础"),
        }
    except Exception as e:
        if MATH_DEBUG:
            print(f"Error calling MODEL_LITE classifier ({model_name}): {e}", flush=True)
        return {
            "problem_type": "代数",
            "hierarchy": "高中",
            "difficulty": "基础",
        }


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
    extraction_attempt = 0
    max_extraction_retries = code_extraction_max_retries()

    while True:
        try:
            if pro_limiter:
                pro_limiter.acquire()
            user_prompt = (
                f"请分析并求解以下题目：\n\n{question}"
                if extraction_attempt == 0
                else code_retry_user_prompt(question, final_content)
            )
            response = client.chat.completions.create(
                model=os.getenv("MODEL_PRO", "gemini-3.1-pro-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                stream=False,
                timeout=MATH_MODEL_TIMEOUT,
                **reasoning_effort_kwargs("PRO"),
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

            generated_code = extract_python_code(final_content)
            if not generated_code.strip():
                if extraction_attempt < max_extraction_retries:
                    extraction_attempt += 1
                    continue
                generated_code = "raise RuntimeError('No executable Python code was extracted from the model output after retry.')"
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


_docker_client = None


def _get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env(timeout=120)
    else:
        try:
            _docker_client.ping()
        except Exception:
            _docker_client = docker.from_env(timeout=120)
    return _docker_client


def _docker_volumes(env_mode="rw", include_pip_cache=True):
    volumes = {
        MATH_CONDA_ENVS_VOLUME: {"bind": "/opt/conda/envs", "mode": env_mode},
    }
    if include_pip_cache:
        volumes[MATH_CONDA_PIP_CACHE_VOLUME] = {"bind": "/root/.cache/pip", "mode": "rw"}
    return volumes


def _run_conda_container(
    command,
    timeout,
    network_disabled=False,
    mem_limit=None,
    env_mode="rw",
    include_pip_cache=True,
):
    container = _get_docker_client().containers.run(
        image=MATH_CONDA_BASE_IMAGE,
        command=command,
        detach=True,
        volumes=_docker_volumes(env_mode=env_mode, include_pip_cache=include_pip_cache),
        environment={
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        },
        network_disabled=network_disabled,
        mem_limit=mem_limit,
    )
    try:
        result = container.wait(timeout=timeout)
        stdout_logs = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr_logs = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        return result.get("StatusCode", 0), stdout_logs, stderr_logs
    except requests.exceptions.ReadTimeout:
        try:
            container.kill()
        finally:
            return 124, "", f"Error: command exceeded the {timeout} seconds timeout."
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def _ensure_conda_math_env():
    with _conda_env_lock:
        env_name = shlex.quote(MATH_CONDA_ENV_NAME)
        check_cmd = ["bash", "-lc", f"test -x /opt/conda/envs/{env_name}/bin/python"]
        status_code, _, _ = _run_conda_container(check_cmd, timeout=60)
        if status_code == 0:
            return

        packages = " ".join(shlex.quote(pkg) for pkg in MATH_CONDA_INITIAL_PACKAGES)
        create_cmd = [
            "bash",
            "-lc",
            (
                f"conda create -y -n {env_name} python=3.10 && "
                f"/opt/conda/envs/{env_name}/bin/python -m pip install {packages}"
            ),
        ]
        status_code, stdout_logs, stderr_logs = _run_conda_container(
            create_cmd,
            timeout=MATH_CONDA_INSTALL_TIMEOUT,
        )
        if status_code != 0:
            output = "\n".join(part for part in (stdout_logs, stderr_logs) if part)
            raise RuntimeError(f"Failed to initialize conda math env: {output}")


def _install_missing_module(module_name: str):
    with _conda_env_lock:
        env_name = shlex.quote(MATH_CONDA_ENV_NAME)
        package = shlex.quote(module_name)
        cmd = [
            "bash",
            "-lc",
            f"/opt/conda/envs/{env_name}/bin/python -m pip install {package}",
        ]
        status_code, stdout_logs, stderr_logs = _run_conda_container(
            cmd,
            timeout=MATH_CONDA_INSTALL_TIMEOUT,
        )
        if status_code != 0:
            output = "\n".join(part for part in (stdout_logs, stderr_logs) if part)
            raise RuntimeError(f"Failed to install missing module {module_name}: {output}")


def _resolve_wolfram_script_path():
    if WOLFRAM_SCRIPT_PATH:
        return WOLFRAM_SCRIPT_PATH
    found = shutil.which("wolframscript")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Wolfram Research\Wolfram Engine\14.3\wolframscript.exe",
        r"C:\Program Files\Wolfram Research\WolframScript\wolframscript.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "wolframscript"


class _WolframBridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/eval":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if payload.get("token") != WOLFRAM_BRIDGE_TOKEN:
                self.send_error(403)
                return

            expr = str(payload.get("expr", ""))
            timeout = int(payload.get("timeout", 30))
            timeout = max(1, min(timeout, 120))

            with _wolfram_process_lock:
                completed = subprocess.run(
                    [_resolve_wolfram_script_path(), "-code", expr],
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )

            body = json.dumps(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except subprocess.TimeoutExpired:
            self._send_json({"returncode": 124, "stdout": "", "stderr": "Wolfram evaluation timed out"})
        except Exception as e:
            self._send_json({"returncode": 1, "stdout": "", "stderr": str(e)})

    def log_message(self, format, *args):
        if MATH_DEBUG:
            super().log_message(format, *args)

    def _send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _ensure_wolfram_bridge():
    global _wolfram_bridge_server, WOLFRAM_BRIDGE_PORT
    if not WOLFRAM_BRIDGE_ENABLED:
        return

    with _wolfram_bridge_lock:
        if _wolfram_bridge_server is not None:
            return
        try:
            _wolfram_bridge_server = ThreadingHTTPServer(
                (WOLFRAM_BRIDGE_HOST, WOLFRAM_BRIDGE_PORT),
                _WolframBridgeHandler,
            )
        except OSError:
            _wolfram_bridge_server = ThreadingHTTPServer(
                (WOLFRAM_BRIDGE_HOST, 0),
                _WolframBridgeHandler,
            )
            WOLFRAM_BRIDGE_PORT = _wolfram_bridge_server.server_address[1]
        thread = threading.Thread(target=_wolfram_bridge_server.serve_forever, daemon=True)
        thread.start()


def _wolfram_python_prelude():
    if not WOLFRAM_BRIDGE_ENABLED:
        return ""
    return f'''
import json as _lg_wolfram_json
import sys as _lg_wolfram_sys
import types as _lg_wolfram_types
import urllib.request as _lg_wolfram_urllib_request

def wolfram_eval(expr, timeout=30):
    payload = _lg_wolfram_json.dumps({{
        "token": {WOLFRAM_BRIDGE_TOKEN!r},
        "expr": str(expr),
        "timeout": int(timeout),
    }}).encode("utf-8")
    req = _lg_wolfram_urllib_request.Request(
        "http://{WOLFRAM_BRIDGE_CONNECT_HOST}:{WOLFRAM_BRIDGE_PORT}/eval",
        data=payload,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with _lg_wolfram_urllib_request.urlopen(req, timeout=timeout + 5) as resp:
        result = _lg_wolfram_json.loads(resp.read().decode("utf-8"))
    if result.get("returncode") != 0:
        raise RuntimeError(result.get("stderr") or result.get("stdout") or "Wolfram evaluation failed")
    return result.get("stdout", "").strip()

class _LgWolframEvalModule(_lg_wolfram_types.ModuleType):
    def __call__(self, expr, timeout=30):
        return wolfram_eval(expr, timeout=timeout)

_lg_wolfram_module = _LgWolframEvalModule("wolfram_eval")
_lg_wolfram_module.wolfram_eval = wolfram_eval
_lg_wolfram_sys.modules["wolfram_eval"] = _lg_wolfram_module
'''


def math_conda_code_executor_node(state: GraphState):
    generated_code = state.get("generated_code", "")
    if not generated_code.strip():
        return {"execution_output": "Error: No code to execute."}

    try:
        _ensure_conda_math_env()
        _ensure_wolfram_bridge()
        max_retries = 3
        execution_output = ""
        code_to_run = _wolfram_python_prelude() + "\n" + generated_code

        for attempt in range(max_retries):
            run_cmd = [
                f"/opt/conda/envs/{MATH_CONDA_ENV_NAME}/bin/python",
                "-c",
                code_to_run,
            ]
            status_code, stdout_logs, stderr_logs = _run_conda_container(
                run_cmd,
                timeout=MATH_CODE_TIMEOUT,
                network_disabled=not WOLFRAM_BRIDGE_ENABLED,
                mem_limit="1g",
                env_mode="ro",
                include_pip_cache=False,
            )

            output_parts = []
            if stdout_logs:
                output_parts.append(f"----- STDOUT -----\n{stdout_logs}")
            if stderr_logs:
                output_parts.append(f"----- STDERR -----\n{stderr_logs}")
            if status_code != 0:
                output_parts.append(f"Exit Code: {status_code}")
            execution_output = "\n".join(output_parts) if output_parts else "Success (No Output)"

            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr_logs)
            if match and attempt < max_retries - 1:
                missing_module = match.group(1)
                if missing_module in MATH_BUILTIN_HELPER_MODULES:
                    break
                _install_missing_module(missing_module)
                continue
            break

        return {"execution_output": execution_output}
    except Exception as e:
        return {"execution_output": f"Execution setup failed: {str(e)}"}


def _quiet_node(fn):
    def wrapped(*args, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)

    return wrapped


def _build_math_graph(debug: bool):
    workflow = StateGraph(GraphState)
    quiet = (lambda fn: fn) if debug else _quiet_node
    solve_node = analyze_and_solve_node if debug else analyze_and_solve_non_stream_node

    workflow.add_node("type_classifier", quiet(type_classifier_lite_node))
    workflow.add_node("analyze_and_solve", quiet(solve_node))
    workflow.add_node("code_executor", quiet(math_conda_code_executor_node))
    workflow.add_node("judge", quiet(judge_node))
    workflow.add_node("human_review", quiet(lambda state: {}))
    workflow.add_edge(START, "type_classifier")
    workflow.add_edge("type_classifier", "analyze_and_solve")
    workflow.add_conditional_edges(
        "analyze_and_solve",
        quiet(route_after_analyze),
        {"end": END, "code_executor": "code_executor"},
    )
    workflow.add_edge("code_executor", "judge")
    workflow.add_conditional_edges(
        "judge",
        quiet(route_after_judge),
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
    for update in graph_app.stream(initial_state, config=config, stream_mode="updates"):
        if sessions[thread_id].get("status") == "cancelled":
            return

        for node_name, node_update in update.items():
            if node_update is None:
                node_update = {}
            sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update}
            sessions[thread_id]["state"].update(node_update)
        _mark_next_nodes(thread_id, config)

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
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=False)
