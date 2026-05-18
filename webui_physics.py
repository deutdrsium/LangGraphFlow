import json
import os
import re as _re
import sys
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

load_dotenv()

PHYSICS_DEBUG = "-debug" in sys.argv or "--debug" in sys.argv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


class PhysicsTaskData(BaseModel):
    task_id: str
    marking: str
    model_answer: str


class RevisionData(BaseModel):
    tasks: list


physics_sessions = {}
revision_queue: list = []


@app.get("/")
async def get_physics_ui_root():
    with open("frontend/physics.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/physics")
async def get_physics_ui():
    with open("frontend/physics.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/physics_task")
async def receive_physics_task(data: PhysicsTaskData):
    """接收油猴脚本推送的物理题 marking + 模型回答，后台异步评分。"""
    physics_sessions[data.task_id] = {
        "status": "running",
        "marking": data.marking,
        "model_answer": data.model_answer,
        "result": None,
        "error": None,
    }
    print(f"\n--- [Physics] 收到物理评分任务: {data.task_id[:8]}... ---", flush=True)

    def run_physics_grading():
        from openai import OpenAI

        phys_client = OpenAI()

        try:
            criteria = []
            raw = data.marking.strip()
            try:
                parsed = json.loads(raw)
                for item in parsed:
                    if isinstance(item, list):
                        criteria.extend(item)
                    elif isinstance(item, str):
                        criteria.append(item)
            except json.JSONDecodeError:
                criteria = [line.strip() for line in raw.splitlines() if line.strip()]

            if not criteria:
                raise ValueError("未能解析出任何评分标准，请检查 marking 字段格式")

            criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))

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
                '  "note": "<≤50 Chinese characters, or null>"\n'
                "}\n\n"
                "IMPORTANT: In score_string, each position must be filled with a number (0 if not awarded, never leave blank).\n"
                "For the note field: set it ONLY when the student solved the problem using a fundamentally different method or approach than what the marking criteria describe (e.g. used energy method instead of Newton's laws). Set to null in all other cases - including partial credit, wrong answers, or minor phrasing differences.\n\n"
                "=========================\n"
                "STRICT RULES FOR matched_text - FOLLOW EXACTLY\n"
                "=========================\n"
                "matched_text is used for EXACT CHARACTER-BY-CHARACTER string search in the student answer.\n"
                "Even one wrong character means total failure. Treat it like a copy-paste operation.\n\n"
                "RULE 1 - VERBATIM COPY ONLY:\n"
                "  Locate the evidence in the student answer. Copy it EXACTLY - every backslash, brace, caret, space, and symbol.\n"
                "  Do NOT reorder, rename, simplify, or reconstruct anything.\n\n"
                "RULE 2 - MATHEMATICAL EQUIVALENCE AND JUMP-CONCATENATION ARE FORBIDDEN:\n"
                "  Even if your version is mathematically equivalent, it is WRONG if it differs from what the student wrote.\n"
                "  Also forbidden: combining the start and end of a formula while skipping the middle.\n\n"
                "  BAD EXAMPLE A (rewrite) - student wrote: `v_{c,m}^2 \\approx 4\\pi G C_m \\left(1 - \\frac{r_m}{r}...\\right)`\n"
                "               you output:   `v_{c,m}(r) \\to \\sqrt{4\\pi G C_m}`   -> INVENTED, NOT IN TEXT, FORBIDDEN\n"
                "  GOOD EXAMPLE A - copy a verbatim prefix/suffix: `v_{c,m}^2 \\approx 4\\pi G C_m`\n\n"
                "  BAD EXAMPLE B (jump-concatenation) - student wrote: `P_i = \\frac{0.2}{...} = ... = 481{,}030\\,\\text{Pa}`\n"
                "               you output:   `P_i = 481{,}030\\,\\text{Pa}`   -> start + end GLUED TOGETHER, NOT CONTIGUOUS, FORBIDDEN\n"
                "  GOOD EXAMPLE B - copy only the final result: `481{,}030 \\, \\text{Pa} \\approx 4.81 \\times 10^5 \\, \\text{Pa}`\n"
                "               OR copy the entire $$ block verbatim.\n\n"
                "RULE 3 - COPY ENTIRE FORMULAS:\n"
                "  If the evidence is inside a $$ ... $$ block, copy the ENTIRE block verbatim from opening $$ to closing $$.\n"
                "  Do not stop in the middle of a formula.\n\n"
                "RULE 4 - NULL IS BETTER THAN WRONG:\n"
                "  If you cannot find the exact text in the student answer, set matched_text to null.\n"
                "  A null is recoverable. An invented string causes a permanent mislabeling.\n\n"
                "RULE 5 - LENGTH:\n"
                "  Aim for 15-80 characters. Long enough to be uniquely locatable, no longer than necessary.\n\n"
                "RULE 6 - DO NOT SET matched_text TO null UNLESS ABSOLUTELY NECESSARY:\n"
                "  A null matched_text means no annotation is placed for that criterion, causing the annotated\n"
                "  score to fall short of the total score. This is almost always wrong.\n"
                "  If two criteria are both satisfied by the same piece of text, return that same text for BOTH -\n"
                "  duplicate matched_text values across criteria are acceptable.\n"
                "  Only use null when the criterion is genuinely not satisfied (satisfied=false) or when no\n"
                "  verbatim evidence exists anywhere in the student answer.\n\n"
                "RULE 7 - CREDIT REQUIRES EXPLICIT EVIDENCE, NOT INFERRED EQUIVALENCE:\n"
                "  Award credit only if the required result is explicitly present in the student answer.\n"
                "  ACCEPTABLE: trivial rearrangement (F=ma -> a=F/m), different but equivalent notation.\n"
                "  NOT ACCEPTABLE: multi-step physical reasoning to bridge what the student wrote to what\n"
                "  EXAMPLE - criterion: 'gives g(r) = G M_int(r)/r^2 via Gauss's theorem or equivalent'\n"
                "    Student writes: v_c(r) = sqrt(G M_b / r)  -> Keplerian law, no Gauss theorem shown\n"
                "    WRONG: 'v_c^2/r = G M_b/r^2 is equivalent to g(r), so satisfied = true'\n"
                "    RIGHT: g(r) = G M_int(r)/r^2 is not explicitly present -> satisfied = false\n"
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
                print(
                    f"[Physics][DBG] system_prompt={len(system_prompt)} chars user_content={len(user_content)} chars",
                    flush=True,
                )
            print(f"[Physics] 调用模型 ({model_name})...", flush=True)

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
                            {"role": "user", "content": user_content},
                        ],
                        temperature=1,
                        stream=True,
                        timeout=_TIMEOUT_SEC,
                    ) as stream:
                        for chunk in stream:
                            if first_chunk:
                                if PHYSICS_DEBUG:
                                    print(
                                        f"[Physics][DBG] 首个chunk，耗时={_time.time() - t_start:.1f}s",
                                        flush=True,
                                    )
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
                    break
                except Exception as api_err:
                    if PHYSICS_DEBUG:
                        print(
                            f"\n[Physics][DBG] 第{_attempt}次调用失败: {type(api_err).__name__}: {api_err}",
                            flush=True,
                        )
                    print(f"[Physics] 调用失败，3秒后重试（第{_attempt + 1}次）...", flush=True)
                    _time.sleep(3)

            content = (content or "").strip()
            print(f"[Physics] 原始响应长度={len(content)}字", flush=True)
            if PHYSICS_DEBUG:
                print(f"[Physics][DBG] 响应内容:\n{'-' * 60}\n{content}\n{'-' * 60}", flush=True)
                print(f"[Physics][DBG] 原始内容 repr（前300字）: {repr(content[:300])}", flush=True)

            content = _re.sub(r"^```(?:json)?\s*", "", content, flags=_re.MULTILINE)
            content = _re.sub(r"\s*```$", "", content, flags=_re.MULTILINE)
            content = content.strip()

            content_before = content
            content = _re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", content)
            if PHYSICS_DEBUG and content != content_before:
                print(
                    f"[Physics][DBG] regex 修正了转义序列，处理后 repr（前300字）: {repr(content[:300])}",
                    flush=True,
                )

            try:
                result = json.loads(content)
            except json.JSONDecodeError as jde:
                print(f"[Physics] json.loads 失败: {jde}", flush=True)
                if PHYSICS_DEBUG:
                    ctx_s = max(0, jde.pos - 60)
                    ctx_e = min(len(content), jde.pos + 60)
                    print(
                        f"[Physics][DBG] 出错位置 pos={jde.pos} 前后内容 repr: {repr(content[ctx_s:ctx_e])}",
                        flush=True,
                    )
                raise

            def _restore_latex_escapes(obj):
                if isinstance(obj, dict):
                    return {k: _restore_latex_escapes(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_restore_latex_escapes(v) for v in obj]
                if isinstance(obj, str):
                    return (
                        obj.replace("\f", "\\f")
                        .replace("\b", "\\b")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                    )
                return obj

            result = _restore_latex_escapes(result)

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
    """油猴脚本轮询：获取指定任务的评分结果。"""
    if task_id not in physics_sessions:
        return {"status": "not_found"}
    session = physics_sessions[task_id]
    if session["status"] == "finished":
        return {"status": "finished", "data": session["result"]}
    if session["status"] == "error":
        return {"status": "error", "error": session.get("error", "Unknown error")}
    return {"status": "running"}


@app.get("/api/physics_sessions")
async def get_physics_sessions():
    """WebUI 获取所有物理评分会话，最多返回最近 50 条。"""
    result = []
    for task_id, session in physics_sessions.items():
        result.append(
            {
                "task_id": task_id,
                "status": session["status"],
                "result": session.get("result"),
                "error": session.get("error"),
            }
        )
    return {"sessions": result[-50:]}


@app.post("/api/physics_clear")
async def clear_physics_sessions():
    """WebUI 清空所有物理评分记录。"""
    physics_sessions.clear()
    return {"status": "success"}


@app.post("/api/physics_revision")
async def submit_revision(data: RevisionData):
    """油猴脚本推送标注失败的题目到返修队列。"""
    for item in data.tasks:
        revision_queue.append(item)
    print(f"[Revision] 收到 {len(data.tasks)} 条返修请求，队列总计 {len(revision_queue)} 条", flush=True)
    return {"status": "success", "count": len(data.tasks), "total": len(revision_queue)}


@app.get("/api/physics_revision")
async def get_revision_queue():
    """查看当前返修队列。"""
    return {"queue": revision_queue, "total": len(revision_queue)}


@app.delete("/api/physics_revision")
async def clear_revision_queue():
    """清空返修队列。"""
    revision_queue.clear()
    return {"status": "success"}


if __name__ == "__main__":
    if PHYSICS_DEBUG:
        print("[WebUI-Physics] DEBUG mode enabled.", flush=True)
    sys.argv = [a for a in sys.argv if a not in ("--debug", "-debug")]
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=False)
