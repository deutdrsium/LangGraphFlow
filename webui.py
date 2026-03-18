from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import json
import time
import threading
from collections import deque

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

@app.get("/")
async def get_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/task_data")
async def receive_task_data(data: TaskData):
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
    return {"sessions": active[:8]}

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
            sessions[thread_id]["nodes"]["trap_check"] = {"status": "executing", "start_time": time.time()}
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
                    # 计算耗时
                    node_info = sessions[thread_id]["nodes"].get(node_name, {})
                    start_time = node_info.get("start_time")
                    duration = round(time.time() - start_time, 1) if start_time else None
                    sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update, "duration": duration}
                    sessions[thread_id]["state"].update(node_update)

                # 获取接下来要执行的节点，将其状态设置为 executing
                current_state = graph_app.get_state(config)
                for next_node in current_state.next:
                    if next_node == "human_review":
                        continue  # human_review 会在循环结束后判断是否 blocked
                    if next_node not in sessions[thread_id]["nodes"]:
                        sessions[thread_id]["nodes"][next_node] = {"status": "executing", "start_time": time.time()}
                    elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                        sessions[thread_id]["nodes"][next_node]["status"] = "executing"
                        if not sessions[thread_id]["nodes"][next_node].get("start_time"):
                            sessions[thread_id]["nodes"][next_node]["start_time"] = time.time()
            
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
        if "solve" in sessions[thread_id]["nodes"]:
            node_info = sessions[thread_id]["nodes"]["solve"]
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
                    sessions[thread_id]["nodes"][next_node] = {"status": "executing", "start_time": time.time()}
                elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                    sessions[thread_id]["nodes"][next_node]["status"] = "executing"
                    if not sessions[thread_id]["nodes"][next_node].get("start_time"):
                        sessions[thread_id]["nodes"][next_node]["start_time"] = time.time()

            for update in graph_app.stream(None, config=config, stream_mode="updates"):
                if sessions[thread_id].get("status") == "cancelled":
                    return

                for node_name, node_update in update.items():
                    if node_update is None:
                        node_update = {}
                    node_info = sessions[thread_id]["nodes"].get(node_name, {})
                    start_time = node_info.get("start_time")
                    duration = round(time.time() - start_time, 1) if start_time else None
                    sessions[thread_id]["nodes"][node_name] = {"status": "success", "data": node_update, "duration": duration}
                    sessions[thread_id]["state"].update(node_update)

                current_state = graph_app.get_state(config)
                for next_node in current_state.next:
                    if next_node == "human_review":
                        continue
                    if next_node not in sessions[thread_id]["nodes"]:
                        sessions[thread_id]["nodes"][next_node] = {"status": "executing", "start_time": time.time()}
                    elif sessions[thread_id]["nodes"][next_node].get("status") != "success":
                        sessions[thread_id]["nodes"][next_node]["status"] = "executing"
                        if not sessions[thread_id]["nodes"][next_node].get("start_time"):
                            sessions[thread_id]["nodes"][next_node]["start_time"] = time.time()
            
            sessions[thread_id]["status"] = "finished"
            sessions[thread_id]["nodes"]["__end__"] = {"status": "success", "data": sessions[thread_id]["state"]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            sessions[thread_id]["status"] = "error"
            sessions[thread_id]["error"] = str(e)
            
    threading.Thread(target=resume_graph, daemon=True).start()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=False)
