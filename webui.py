from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import json
import re as _re
import threading
import sys
from collections import deque
from dotenv import load_dotenv
import os
from rate_limiter import instance_limiter

load_dotenv()

# 调试开关：python webui.py -debug 时开启
# 开启后输出模型响应全文、repr、chunk 等详细内容；默认只保留标识性行
PHYSICS_DEBUG = "-debug" in sys.argv

MAX_INSTANCES = min(int(os.getenv("MAX_INSTANCES", "8")), 8)
if MAX_INSTANCES < 1:
    MAX_INSTANCES = 1

# 导入 main.py 中的相关图定义与变量
from main import graph_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 提供 frontend 静态文件
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
    decision: str  # e.g., "Manual_Confirmed_Match" or "Manual_Overruled_Mismatch"

class PhysicsTaskData(BaseModel):
    task_id: str
    marking: str   # JSON 字符串，如 '[["Award 0.2 pt if...", "Award 0.3 pt if..."]]'
    model_answer: str

# 物理评分会话存储（task_id -> session dict）
physics_sessions = {}

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
        print(f"[Rate Limit] Task {data.task_id} waiting for INSTANCE slot...", flush=True)
        await instance_limiter.async_acquire()
    print(f"\n--- New Data Received! ---")
    print(f"Task ID: {data.task_id}")
    print(f"Question Length: {len(data.question_content)} chars")
    # print(f"Question Preview: {data.question_content[:150]}...")
    # print(f"Answer Preview: {data.answer[:150]}...")
    print("--------------------------\n")
    task_queue.append(data)
    return {"status": "success", "message": "Data queued successfully"}

@app.post("/api/cancel_task")
async def cancel_task(data: CancelData):
    print(f"\n--- Cancel Task Request Received ---")
    print(f"Canceling Task ID: {data.task_id}")
    
    # 清理可能还在队列中未领取的任务
    global task_queue
    original_len = len(task_queue)
    task_queue = deque([task for task in task_queue if task.task_id != data.task_id])
    if len(task_queue) < original_len:
        print(f"Task {data.task_id} cleared from queue")

    # 中断并清理对应 session 的执行
    found = False
    for thread_id, session in sessions.items():
        if session.get("state", {}).get("question_id") == data.task_id:
            session["status"] = "cancelled"
            print(f"Task {data.task_id} map to thread {thread_id} set to cancelled")
            found = True
            
    if not found:
         print(f"Warning: No valid session running for Task {data.task_id}")
         
    return {"status": "success", "message": "Task cancelled"}

@app.get("/api/get_task")
async def get_task():
    if task_queue:
        task = task_queue.popleft()
        return {"has_task": True, "task": task.dict()}
    return {"has_task": False}

@app.get("/api/active_sessions")
async def get_active_sessions():
    active = []
    for tid, session in sessions.items():
        if session.get("status") not in ["cancelled"]:
            active.append({
                "thread_id": tid,
                "task_id": session.get("state", {}).get("question_id", ""),
                "question": session.get("state", {}).get("question_context", ""),
                "truth": session.get("state", {}).get("ground_truth", ""),
                "status": session.get("status"),
                "nodes": session.get("nodes", {})
            })
    return {"sessions": active[:MAX_INSTANCES]}

@app.get("/api/task_result/{task_id}")
async def get_task_result(task_id: str):
    # Iterate through all sessions to find the one matching task_id
    for thread_id, session in sessions.items():
        if session.get("state", {}).get("question_id") == task_id:
            if session.get("status") == "finished":
                return {"status": "finished", "data": session.get("state")}
            elif session.get("status") == "blocked":
                # For HITL, consider it finished from the script's perspective
                state_data = session.get("state").copy()
                state_data["final_decision"] = "HITL"
                state_data["hitl_reason"] = f"系统置信度低 ({state_data.get('confidence_score', 0)} < 75)，需要人工介入审查"
                state_data["code_execution_result"] = state_data.get("execution_output", "")
                return {"status": "finished", "data": state_data}
            elif session.get("status") == "running":
                return {"status": "running"}
    return {"status": "not_found"}

@app.post("/api/run")
async def start_run(data: RunData):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "question_id": data.task_id if data.task_id else f"q_{thread_id[:4]}",
        "question_context": data.question,
        "ground_truth": data.truth
    }
    
    sessions[thread_id] = {
        "status": "running",
        "nodes": {
            "__start__": {"status": "success", "data": initial_state}
        },
        "state": initial_state
    }
    
    def run_graph():
        print(f"\n[WebUI] 收到新请求开始图构建: {data.question[:20]}...", flush=True)
        try:
            sessions[thread_id]["nodes"]["type_classifier"] = {"status": "executing"}
            # 通过 generator 一步步执行
            for update in graph_app.stream(initial_state, config=config, stream_mode="updates"):
                # 如果从外部被取消了
                if sessions[thread_id].get("status") == "cancelled":
                    print(f"\n[WebUI] Process for thread {thread_id} was cancelled mid-flight.", flush=True)
                    return
                
                for node_name, node_update in update.items():
                    print(f"\n[WebUI] 节点 '{node_name}' 执行完成.", flush=True)
                    if node_update is None:
                        node_update = {}
                    sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update}
                    sessions[thread_id]["state"].update(node_update)
                
                # 获取接下来要执行的节点，将其状态设置为 executing
                current_state = graph_app.get_state(config)
                for next_node in current_state.next:
                    if next_node == "human_review":
                        continue  # human_review 会在循环结束后判断是否 blocked
                    if next_node not in sessions[thread_id]["nodes"]:
                        sessions[thread_id]["nodes"][next_node] = {"status": "executing"}
                    elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                        sessions[thread_id]["nodes"][next_node]["status"] = "executing"
            
            # 判断是否被人工审核打断
            state_info = graph_app.get_state(config)
            needs_hitl = state_info.next and "human_review" in state_info.next
            if needs_hitl:
                print(f"\n[WebUI] 流程中断，等待人工审查 (HITL)...", flush=True)
                sessions[thread_id]["status"] = "blocked"
                sessions[thread_id]["nodes"]["human_review"] = {"status": "blocked", "data": {"message": "Waiting for manual review..."}}
            else:
                print(f"\n[WebUI] 流程执行完毕，到达 END.", flush=True)
                sessions[thread_id]["status"] = "finished"
                sessions[thread_id]["nodes"]["__end__"] = {"status": "success", "data": sessions[thread_id]["state"]}

        except Exception as e:
            print(f"\n[WebUI] 执行过程中发生错误: {e}", flush=True)
            sessions[thread_id]["status"] = "error"
            sessions[thread_id]["error"] = str(e)
            # 将当前正在执行的节点标记为 error，便于前端红色标注
            for node_name, node_info in sessions[thread_id]["nodes"].items():
                if isinstance(node_info, dict) and node_info.get("status") == "executing":
                    node_info["status"] = "error"
                    node_info["data"] = {"error": str(e)}
            
    threading.Thread(target=run_graph, daemon=True).start()
    return {"thread_id": thread_id}

@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str):
    if thread_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
        
    import main
    if thread_id in main.streaming_store:
        # 如果代码生成器节点尚未写入最终结果，则向其中注入当前的流式文本
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
    
    # 更新 state
    graph_app.update_state(config, {"final_decision": data.decision})
    
    session["status"] = "running"
    session["nodes"]["human_review"] = {"status": "success", "data": {"final_decision": data.decision}}
    
    def resume_graph():
        try:
            current_state = graph_app.get_state(config)
            for next_node in current_state.next:
                if next_node not in sessions[thread_id]["nodes"]:
                    sessions[thread_id]["nodes"][next_node] = {"status": "executing"}
                elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                    sessions[thread_id]["nodes"][next_node]["status"] = "executing"

            for update in graph_app.stream(None, config=config, stream_mode="updates"):
                if sessions[thread_id].get("status") == "cancelled":
                    return

                for node_name, node_update in update.items():
                    if node_update is None:
                        node_update = {}
                    sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update}
                    sessions[thread_id]["state"].update(node_update)
                
                current_state = graph_app.get_state(config)
                for next_node in current_state.next:
                    if next_node == "human_review":
                        continue
                    if next_node not in sessions[thread_id]["nodes"]:
                        sessions[thread_id]["nodes"][next_node] = {"status": "executing"}
                    elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                        sessions[thread_id]["nodes"][next_node]["status"] = "executing"
            
            sessions[thread_id]["status"] = "finished"
            sessions[thread_id]["nodes"]["__end__"] = {"status": "success", "data": sessions[thread_id]["state"]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            sessions[thread_id]["status"] = "error"
            sessions[thread_id]["error"] = str(e)
            
    threading.Thread(target=resume_graph, daemon=True).start()
    return {"status": "success"}

# ==========================================
# 物理评分接口
# ==========================================

@app.get("/physics")
async def get_physics_ui():
    with open("frontend/physics.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/physics_task")
async def receive_physics_task(data: PhysicsTaskData):
    """接收油猴脚本推送的物理题 marking + 模型回答，后台异步评分"""
    physics_sessions[data.task_id] = {
        "status": "running",
        "marking": data.marking,
        "model_answer": data.model_answer,
        "result": None,
        "error": None
    }
    print(f"\n--- [Physics] 收到物理评分任务: {data.task_id[:8]}... ---", flush=True)

    def run_physics_grading():
        from openai import OpenAI
        phys_client = OpenAI()

        try:
            # 解析 marking 标准
            criteria = []
            raw = data.marking.strip()
            try:
                parsed = json.loads(raw)
                # 支持 [["..."]] 或 ["..."] 两种嵌套格式
                for item in parsed:
                    if isinstance(item, list):
                        criteria.extend(item)
                    elif isinstance(item, str):
                        criteria.append(item)
            except json.JSONDecodeError:
                # 降级：把每行当作一条评分标准
                criteria = [line.strip() for line in raw.splitlines() if line.strip()]

            if not criteria:
                raise ValueError("未能解析出任何评分标准，请检查 marking 字段格式")

            criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))

            system_prompt = (
                "You are a physics exam grader. Grade the student's answer strictly based on the provided marking criteria.\n\n"
                "For each criterion extract the point value from 'Award X pt if ...' or 'Award X marks if ...' patterns.\n"
                "Determine whether the student's answer satisfies each criterion.\n\n"
                "Return ONLY a valid JSON object (no markdown, no code blocks) with this exact structure:\n"
                "{\n"
                '  "results": [\n'
                '    {\n'
                '      "criterion": "<full criterion text>",\n'
                '      "points_awarded": <number, 0 if not satisfied>,\n'
                '      "satisfied": <true or false>,\n'
                '      "matched_text": "<VERBATIM character-for-character copy from student answer, or null>"\n'
                "    }\n"
                "  ],\n"
                '  "total_score": <sum of points_awarded>,\n'
                '  "score_string": "<awarded1>+<awarded2>+...=<total>, e.g. 0.2+0+0.3=0.5>",\n'
                '  "note": "<≤20 Chinese characters, or null>"\n'
                "}\n\n"
                "IMPORTANT: In score_string, each position must be filled with a number (0 if not awarded, never leave blank).\n"
                "For the note field: set it ONLY when the student solved the problem using a fundamentally different method or approach than what the marking criteria describe (e.g. used energy method instead of Newton's laws). Set to null in all other cases — including partial credit, wrong answers, or minor phrasing differences.\n\n"
                "══════════════════════════════════════════════\n"
                "STRICT RULES FOR matched_text — FOLLOW EXACTLY\n"
                "══════════════════════════════════════════════\n"
                "matched_text is used for EXACT CHARACTER-BY-CHARACTER string search in the student answer.\n"
                "Even one wrong character means total failure. Treat it like a copy-paste operation.\n\n"
                "RULE 1 — VERBATIM COPY ONLY:\n"
                "  Locate the evidence in the student answer. Copy it EXACTLY — every backslash, brace, caret, space, and symbol.\n"
                "  Do NOT reorder, rename, simplify, or reconstruct anything.\n\n"
                "RULE 2 — MATHEMATICAL EQUIVALENCE AND JUMP-CONCATENATION ARE FORBIDDEN:\n"
                "  Even if your version is mathematically equivalent, it is WRONG if it differs from what the student wrote.\n"
                "  Also forbidden: combining the start and end of a formula while skipping the middle.\n\n"
                "  BAD EXAMPLE A (rewrite) — student wrote: `v_{c,m}^2 \\approx 4\\pi G C_m \\left(1 - \\frac{r_m}{r}...\\right)`\n"
                "               you output:   `v_{c,m}(r) \\to \\sqrt{4\\pi G C_m}`   ← INVENTED, NOT IN TEXT, FORBIDDEN\n"
                "  GOOD EXAMPLE A — copy a verbatim prefix/suffix: `v_{c,m}^2 \\approx 4\\pi G C_m`\n\n"
                "  BAD EXAMPLE B (jump-concatenation) — student wrote: `P_i = \\frac{0.2}{...} = ... = 481{,}030\\,\\text{Pa}`\n"
                "               you output:   `P_i = 481{,}030\\,\\text{Pa}`   ← start + end GLUED TOGETHER, NOT CONTIGUOUS, FORBIDDEN\n"
                "  GOOD EXAMPLE B — copy only the final result: `481{,}030 \\, \\text{Pa} \\approx 4.81 \\times 10^5 \\, \\text{Pa}`\n"
                "               OR copy the entire $$ block verbatim.\n\n"
                "RULE 3 — COPY ENTIRE FORMULAS:\n"
                "  If the evidence is inside a $$ ... $$ block, copy the ENTIRE block verbatim from opening $$ to closing $$.\n"
                "  Do not stop in the middle of a formula.\n\n"
                "RULE 4 — NULL IS BETTER THAN WRONG:\n"
                "  If you cannot find the exact text in the student answer, set matched_text to null.\n"
                "  A null is recoverable. An invented string causes a permanent mislabeling.\n\n"
                "RULE 5 — LENGTH:\n"
                "  Aim for 15–80 characters. Long enough to be uniquely locatable, no longer than necessary.\n\n"
                "RULE 6 — DO NOT SET matched_text TO null UNLESS ABSOLUTELY NECESSARY:\n"
                "  A null matched_text means no annotation is placed for that criterion, causing the annotated\n"
                "  score to fall short of the total score. This is almost always wrong.\n"
                "  If two criteria are both satisfied by the same piece of text, return that same text for BOTH —\n"
                "  duplicate matched_text values across criteria are acceptable.\n"
                "  Only use null when the criterion is genuinely not satisfied (satisfied=false) or when no\n"
                "  verbatim evidence exists anywhere in the student answer.\n\n"
                "RULE 7 — CREDIT REQUIRES EXPLICIT EVIDENCE, NOT INFERRED EQUIVALENCE:\n"
                "  Award credit only if the required result is explicitly present in the student answer.\n"
                "  ACCEPTABLE: trivial rearrangement (F=ma ↔ a=F/m), different but equivalent notation.\n"
                "  NOT ACCEPTABLE: multi-step physical reasoning to bridge what the student wrote to what\n"
                "  EXAMPLE — criterion: 'gives g(r) = G M_int(r)/r² via Gauss's theorem or equivalent'\n"
                "    Student writes: v_c(r) = sqrt(G M_b / r)  ← Keplerian law, no Gauss theorem shown\n"
                "    WRONG: 'v_c²/r = G M_b/r² is equivalent to g(r), so satisfied = true'\n"
                "    RIGHT: g(r) = G M_int(r)/r² is not explicitly present → satisfied = false\n"
                "  The formula the criterion asks for must appear in the student's own text."
            )

            user_content = (
                f"=== Marking Criteria ===\n{criteria_text}\n\n"
                f"=== Student Answer ===\n{data.model_answer}"
            )

            from rate_limiter import flash_limiter
            if flash_limiter:
                print("[Rate Limit] Waiting for MODEL_FLASH slot (physics)...", flush=True)
                flash_limiter.acquire()

            import time as _time
            model_name = os.getenv("MODEL_FLASH", "gemini-3-flash-preview")
            if PHYSICS_DEBUG:
                print(f"[Physics][DBG] base_url={phys_client.base_url}  model={model_name}", flush=True)
                print(f"[Physics][DBG] system_prompt={len(system_prompt)}字  user_content={len(user_content)}字", flush=True)
            print(f"[Physics] 调用模型 ({model_name})...", flush=True)

            # 无限重试，每次超时限制 120 秒
            _TIMEOUT_SEC = 120
            content = None
            _attempt = 0

            while True:
                _attempt += 1
                print(f"[Physics] 第{_attempt}次流式调用（timeout={_TIMEOUT_SEC}s）...", flush=True)
                try:
                    t_start = _time.time()
                    content_parts = []
                    first_chunk = True
                    with phys_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=1,
                        stream=True,
                        timeout=_TIMEOUT_SEC
                    ) as stream:
                        for chunk in stream:
                            if first_chunk:
                                if PHYSICS_DEBUG:
                                    print(f"[Physics][DBG] 首个chunk，耗时={_time.time()-t_start:.1f}s", flush=True)
                                first_chunk = False
                            delta = chunk.choices[0].delta.content if chunk.choices else None
                            if delta:
                                if PHYSICS_DEBUG:
                                    print(delta, end="", flush=True)
                                content_parts.append(delta)

                    elapsed = _time.time() - t_start
                    total_chars = sum(len(p) for p in content_parts)
                    print(f"\n[Physics] 流结束，耗时={elapsed:.1f}s，{total_chars}字", flush=True)
                    content = "".join(content_parts)
                    break  # 成功，退出重试循环
                except Exception as api_err:
                    if PHYSICS_DEBUG:
                        print(f"\n[Physics][DBG] 第{_attempt}次调用失败: {type(api_err).__name__}: {api_err}", flush=True)
                    print(f"[Physics] 调用失败，3秒后重试（第{_attempt + 1}次）...", flush=True)
                    _time.sleep(3)

            content = content or ""
            content = content.strip()
            print(f"[Physics] 原始响应长度={len(content)}字", flush=True)
            if PHYSICS_DEBUG:
                print(f"[Physics][DBG] 响应内容：\n{'─'*60}\n{content}\n{'─'*60}", flush=True)
                # 用 repr 打出原始字节，避免终端渲染 \r\n 等控制字符造成误判
                print(f"[Physics][DBG] 原始内容 repr（前300字）: {repr(content[:300])}", flush=True)

            # 移除可能的 markdown 代码块
            content = _re.sub(r'^```(?:json)?\s*', '', content, flags=_re.MULTILINE)
            content = _re.sub(r'\s*```$', '', content, flags=_re.MULTILINE)
            content = content.strip()

            # 修复模型输出的非法 JSON 转义序列（如 \p \a \m 等 LaTeX 命令）
            # JSON 合法转义：\" \\ \/ \b \f \n \r \t \uXXXX，其余均需补全为 \\
            # (?<!\\) 负向前瞻：跳过已被转义的反斜杠（即 \\ 的第二个 \），避免把合法的 \\pi 破坏成 \\\pi
            content_before = content
            content = _re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', content)
            if PHYSICS_DEBUG and content != content_before:
                print(f"[Physics][DBG] regex 修正了转义序列，处理后 repr（前300字）: {repr(content[:300])}", flush=True)

            try:
                result = json.loads(content)
            except json.JSONDecodeError as jde:
                print(f"[Physics] json.loads 失败: {jde}", flush=True)
                if PHYSICS_DEBUG:
                    ctx_s = max(0, jde.pos - 60)
                    ctx_e = min(len(content), jde.pos + 60)
                    print(f"[Physics][DBG] 出错位置 pos={jde.pos} 前后内容 repr: {repr(content[ctx_s:ctx_e])}", flush=True)
                raise

            # Post-fix: LLM often omits the double-backslash in LaTeX commands,
            # e.g. outputs \frac instead of \\frac inside JSON strings.
            # json.loads() silently converts \f → form-feed (U+000C),
            # \b → backspace (U+0008), \r → CR (U+000D), \t → tab (U+0009).
            # These control characters are meaningless in LaTeX math text —
            # restore them to their literal backslash+letter form.
            def _restore_latex_escapes(obj):
                if isinstance(obj, dict):
                    return {k: _restore_latex_escapes(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_restore_latex_escapes(v) for v in obj]
                if isinstance(obj, str):
                    return (obj
                            .replace('\f', '\\f')   # \frac \forall \footnote ...
                            .replace('\b', '\\b')   # \begin \beta \bf ...
                            .replace('\r', '\\r')   # \rho \right \rm ...
                            .replace('\t', '\\t'))  # \theta \tau \text \times ...
                return obj
            result = _restore_latex_escapes(result)

            # 确保 score_string 的每个位置都有数字（0而非空）
            if "results" in result:
                parts = [str(r.get("points_awarded", 0)) for r in result["results"]]
                total = sum(r.get("points_awarded", 0) for r in result["results"])
                result["score_string"] = "+".join(parts) + "=" + str(round(total, 4))
                result["total_score"] = round(total, 4)

            physics_sessions[data.task_id]["status"] = "finished"
            physics_sessions[data.task_id]["result"] = result
            print(f"[Physics] 评分完成: {data.task_id[:8]}... 总分={result.get('total_score')}", flush=True)

        except Exception as e:
            print(f"[Physics] 评分出错: {e}", flush=True)
            physics_sessions[data.task_id]["status"] = "error"
            physics_sessions[data.task_id]["error"] = str(e)

    threading.Thread(target=run_physics_grading, daemon=True).start()
    return {"status": "success", "message": "Physics task queued"}

@app.get("/api/physics_result/{task_id}")
async def get_physics_result(task_id: str):
    """油猴脚本轮询：获取指定任务的评分结果"""
    if task_id not in physics_sessions:
        return {"status": "not_found"}
    session = physics_sessions[task_id]
    if session["status"] == "finished":
        return {"status": "finished", "data": session["result"]}
    elif session["status"] == "error":
        return {"status": "error", "error": session.get("error", "Unknown error")}
    return {"status": "running"}

@app.get("/api/physics_sessions")
async def get_physics_sessions():
    """WebUI 获取所有物理评分会话（最多返回最近 50 条）"""
    result = []
    for task_id, session in physics_sessions.items():
        result.append({
            "task_id": task_id,
            "status": session["status"],
            "result": session.get("result"),
            "error": session.get("error")
        })
    return {"sessions": result[-50:]}

@app.post("/api/physics_clear")
async def clear_physics_sessions():
    """WebUI 清空所有物理评分记录"""
    physics_sessions.clear()
    return {"status": "success"}


# ==========================================
# 返修队列接口
# ==========================================

class RevisionData(BaseModel):
    tasks: list  # [{processKey, taskId, subQId, failedCriteria, timestamp}]

revision_queue: list = []

@app.post("/api/physics_revision")
async def submit_revision(data: RevisionData):
    """油猴脚本推送标注失败的题目到返修队列"""
    for item in data.tasks:
        revision_queue.append(item)
    print(f"[Revision] 收到 {len(data.tasks)} 条返修请求，队列总计 {len(revision_queue)} 条", flush=True)
    return {"status": "success", "count": len(data.tasks), "total": len(revision_queue)}

@app.get("/api/physics_revision")
async def get_revision_queue():
    """查看当前返修队列"""
    return {"queue": revision_queue, "total": len(revision_queue)}

@app.delete("/api/physics_revision")
async def clear_revision_queue():
    """清空返修队列"""
    revision_queue.clear()
    return {"status": "success"}


if __name__ == "__main__":
    if PHYSICS_DEBUG:
        print("[WebUI] *** DEBUG 模式已开启 ***", flush=True)
    else:
        print("[WebUI] 正常模式（加 -debug 参数可开启详细日志）", flush=True)
    # 将 -debug 从 sys.argv 中移除，避免 uvicorn 误解析该参数
    sys.argv = [a for a in sys.argv if a != "-debug"]
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=False)
