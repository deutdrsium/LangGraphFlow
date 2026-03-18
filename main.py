from typing import TypedDict
import os
import json
import re
import time
import subprocess
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()
client = OpenAI()

# 全局存储流式输出用于前端展示
streaming_store = {}


def stream_chat(model: str, messages: list, temperature: float = 1.0, timeout: float = 60.0) -> str:
    """统一的流式调用封装，返回最终正文内容（不含 reasoning）"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        timeout=timeout,
        reasoning_effort="high"
    )
    content = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)
    print()
    return content


def stream_chat_with_display(model: str, messages: list, thread_id: str = None, temperature: float = 1.0, timeout: float = 60.0) -> str:
    """流式调用封装（带 CoT 展示），返回最终正文内容"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        timeout=timeout,
        reasoning_effort="high"
    )
    display_content = ""
    final_content = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            delta_dict = delta.model_dump()
            reasoning = delta_dict.get("reasoning_content")
            if reasoning:
                display_content += reasoning
                print(reasoning, end="", flush=True)
            if delta.content:
                display_content += delta.content
                final_content += delta.content
                print(delta.content, end="", flush=True)
            if thread_id:
                streaming_store[thread_id] = display_content
    print()
    return final_content


class GraphState(TypedDict):
    question_id: str
    question_context: str
    ground_truth: str
    trap_analysis: bool
    trap_reason: str
    generated_code: str
    execution_output: str
    confidence_score: float
    final_decision: str
    verified_ans: str
    # 二次验证相关
    retry_code: str
    retry_output: str
    retry_decision: str
    retry_verified_ans: str
    retry_confidence: float


def trap_check_node(state: GraphState):
    """Node 1: 可解性检测 - 判断题目是否存在逻辑陷阱或不可解"""
    print("\n---> [Node: trap_check] 开始执行...", flush=True)
    question = state.get("question_context", "")
    prompt = os.getenv("TRAP_CHECK_PROMPT", "You are an expert math problem analyzer.")
    prompt += '\n\nPlease return ONLY a JSON object: {"solvable": true/false, "reason": "<string>"}. If solvable, reason should be "pass". Do not use code blocks.'

    try:
        content = stream_chat(
            model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请判断以下题目是否可解：\n\n{question}"}
            ]
        )
        content = content.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(content)
        solvable = result_dict.get("solvable", True)
        reason = result_dict.get("reason", "pass")
        if not solvable:
            print(f"\n>>> [陷阱检测] 题目不可解: {reason}")
        return {
            "trap_analysis": not solvable,
            "trap_reason": reason if not solvable else "pass"
        }
    except Exception as e:
        print(f"Error in trap_check: {e}")
        # 检测失败时默认可解，交给后续节点处理
        return {
            "trap_analysis": False,
            "trap_reason": "pass"
        }

def solve_node(state: GraphState, config: RunnableConfig = None):
    """Node 2: 代码生成 (使用强模型求解题目)"""
    print("\n---> [Node: solve] 开始执行...", flush=True)
    question = state.get("question_context", "")
    thread_id = config.get("configurable", {}).get("thread_id", None) if config else None
    system_prompt = os.getenv("SOLVE_PROMPT", "You are a helpful math assistant.")

    attempt = 0
    error_codes = []

    while True:
        try:
            final_content = stream_chat_with_display(
                model=os.getenv("MODEL_PRO", "gemini-3.1-pro-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请求解以下题目：\n\n{question}"}
                ],
                thread_id=thread_id
            )

            match = re.search(r'```python\n(.*?)\n```', final_content, re.DOTALL)
            generated_code = match.group(1).strip() if match else final_content
            return {"generated_code": generated_code}

        except Exception as e:
            attempt += 1
            is_429 = (hasattr(e, 'status_code') and e.status_code == 429) or '429' in str(e)
            error_codes.append(429 if is_429 else 0)
            print(f"Error in solve (attempt {attempt}): {e}", flush=True)

            if attempt >= 3:
                if all(code == 429 for code in error_codes[-3:]):
                    print("连续3次触发429限流，等待15s后继续重试...", flush=True)
                    time.sleep(15)
                    attempt = 0
                    error_codes.clear()
                    continue
                else:
                    raise Exception(f"solve 连续失败3次 (非全部429限流)，阻塞: {e}")

def code_executor_node(state: GraphState):
    """Node 4: 子进程代码执行器"""
    print("\n---> [Node: code_executor] 开始执行...", flush=True)
    generated_code = state.get("generated_code", "")
    if not generated_code.strip():
        return {"execution_output": "Error: No code to execute."}

    try:
        # 将代码写入临时文件，避免命令行长度限制和转义问题
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(generated_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"----- STDOUT -----\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"----- STDERR -----\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")

            execution_output = "\n".join(output_parts) if output_parts else "Success (No Output)"

        except subprocess.TimeoutExpired:
            execution_output = "Error: Code execution exceeded the 60 seconds timeout."
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"Code execution error: {e}")
        execution_output = f"Execution failed: {str(e)}"

    return {"execution_output": execution_output}

def judge_node(state: GraphState):
    """Node 5: 裁判 (The Judge)"""
    print("\n---> [Node: judge] 开始执行...", flush=True)
    question = state.get("question_context", "")
    ground_truth = state.get("ground_truth", "")
    execution_output = state.get("execution_output", "")
    
    prompt = os.getenv("JUDGE_PROMPT", "You are an expert mathematical judge.")
    prompt += '\n\nPlease return ONLY a JSON object: {"confidence": <int>, "decision": "Match" or "Mismatch" or "Error", "verified_ans": <string>}. Use "pass" for verified_ans if decision is Match. Do not use code blocks.'

    # 组装完整的判断内容
    user_content = f"""
[Original Question]
{question}

[Ground Truth]
{ground_truth}

[Sandbox Execution Output]
{execution_output}
"""

    try:
        content = stream_chat(
            model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
        )
        content = content.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(content)
        return {
            "confidence_score": float(result_dict.get("confidence", 0)),
            "final_decision": result_dict.get("decision", "Error"),
            "verified_ans": result_dict.get("verified_ans", "pass")
        }
    except Exception as e:
        print(f"Error calling OpenAI API in judge: {e}")
        return {
            "confidence_score": 0.0,
            "final_decision": "Error",
            "verified_ans": "pass"
        }

def retry_solve_node(state: GraphState, config: RunnableConfig = None):
    """Node: 二次验证 - 用不同思路重新求解"""
    print("\n---> [Node: retry_solve] 首次 Mismatch，启动二次独立验证...", flush=True)
    question = state.get("question_context", "")
    first_code = state.get("generated_code", "")
    first_output = state.get("execution_output", "")
    thread_id = config.get("configurable", {}).get("thread_id", None) if config else None
    system_prompt = os.getenv("RETRY_SOLVE_PROMPT", "You are a helpful math assistant.")

    attempt = 0
    error_codes = []

    while True:
        try:
            final_content = stream_chat_with_display(
                model=os.getenv("MODEL_PRO", "gemini-3.1-pro-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请用与之前不同的方法求解以下题目：\n\n{question}\n\n之前的求解代码及其输出如下（仅供参考，请用不同的方法）：\n```python\n{first_code}\n```\n输出：{first_output}"}
                ],
                thread_id=thread_id
            )

            match = re.search(r'```python\n(.*?)\n```', final_content, re.DOTALL)
            retry_code = match.group(1).strip() if match else final_content
            return {"retry_code": retry_code}

        except Exception as e:
            attempt += 1
            is_429 = (hasattr(e, 'status_code') and e.status_code == 429) or '429' in str(e)
            error_codes.append(429 if is_429 else 0)
            print(f"Error in retry_solve (attempt {attempt}): {e}", flush=True)
            if attempt >= 3:
                if all(code == 429 for code in error_codes[-3:]):
                    print("连续3次触发429限流，等待15s后继续重试...", flush=True)
                    time.sleep(15)
                    attempt = 0
                    error_codes.clear()
                    continue
                else:
                    raise Exception(f"retry_solve 连续失败3次: {e}")

def retry_executor_node(state: GraphState):
    """Node: 二次验证 - 执行重新生成的代码"""
    print("\n---> [Node: retry_executor] 开始执行二次验证代码...", flush=True)
    retry_code = state.get("retry_code", "")
    if not retry_code.strip():
        return {"retry_output": "Error: No retry code to execute."}

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(retry_code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=60
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"----- STDOUT -----\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"----- STDERR -----\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit Code: {result.returncode}")
            retry_output = "\n".join(output_parts) if output_parts else "Success (No Output)"
        except subprocess.TimeoutExpired:
            retry_output = "Error: Code execution exceeded the 60 seconds timeout."
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Retry execution error: {e}")
        retry_output = f"Execution failed: {str(e)}"

    return {"retry_output": retry_output}

def retry_judge_node(state: GraphState):
    """Node: 二次裁判 - 综合两次执行结果做最终判定"""
    print("\n---> [Node: retry_judge] 综合两次结果做最终判定...", flush=True)
    question = state.get("question_context", "")
    ground_truth = state.get("ground_truth", "")
    first_output = state.get("execution_output", "")
    retry_output = state.get("retry_output", "")

    prompt = os.getenv("RETRY_JUDGE_PROMPT", os.getenv("JUDGE_PROMPT", "You are an expert mathematical judge."))

    user_content = f"""
[Original Question]
{question}

[Ground Truth]
{ground_truth}

[First Solve Output]
{first_output}

[Second Solve Output (independent method)]
{retry_output}

请综合两次独立求解的结果，判断 Ground Truth 是否正确。
如果两次结果一致且与 Ground Truth 不同，则判定 Mismatch；
如果两次结果不一致，则判定 Error（说明求解不可靠，需人工介入）；
如果至少一次结果与 Ground Truth 一致，则判定 Match。
"""
    prompt += '\n\nPlease return ONLY a JSON object: {"confidence": <int>, "decision": "Match" or "Mismatch" or "Error", "verified_ans": <string>}. Do not use code blocks.'

    try:
        content = stream_chat(
            model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
        )
        content = content.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(content)
        decision = result_dict.get("decision", "Error")
        confidence = float(result_dict.get("confidence", 0))
        verified_ans = result_dict.get("verified_ans", "pass")
        return {
            "retry_confidence": confidence,
            "retry_decision": decision,
            "retry_verified_ans": verified_ans,
            "final_decision": decision,
            "confidence_score": confidence,
            "verified_ans": verified_ans
        }
    except Exception as e:
        print(f"Error in retry_judge: {e}")
        return {
            "retry_confidence": 0.0,
            "retry_decision": "Error",
            "retry_verified_ans": "pass",
            "final_decision": "Error",
            "confidence_score": 0.0,
            "verified_ans": "pass"
        }

def human_review_node(state: GraphState):
    """Node 6: 人工审查节点 (HITL)"""
    # 实际运行中，执行到此节点前会被打断。当恢复执行时，会运行此节点。
    print("--- [HITL] 人工审查节点触发，处理并向下流转 ---")
    return {}

def route_after_trap_check(state: GraphState) -> str:
    """如果发现不可解，直接去结尾；否则进入解题"""
    if state.get("trap_analysis", False) is True:
        print(f"\n>>> [路由] 题目不可解: {state.get('trap_reason')}\n>>> 终止流程，直接输出。")
        return "end"
    return "solve"

def route_after_judge(state: GraphState) -> str:
    """条件路由逻辑：Match直通，Mismatch进二次验证，低置信进HITL"""
    confidence = state.get("confidence_score", 0.0)
    decision = state.get("final_decision", "Error")

    if decision == "Match" and confidence >= 75:
        print(f"\n>>> [路由] Match (置信度 {confidence})，直接输出。")
        return "end"
    elif decision == "Mismatch" and confidence >= 75:
        print(f"\n>>> [路由] Mismatch (置信度 {confidence})，进入二次独立验证...")
        return "retry_solve"
    else:
        print(f"\n>>> [路由] 置信度 {confidence} 不足或 Error，转入人工审查...")
        return "human_review"

def route_after_retry_judge(state: GraphState) -> str:
    """二次裁判后的路由：两次一致则确认，不一致进HITL"""
    decision = state.get("retry_decision", "Error")
    confidence = state.get("retry_confidence", 0.0)

    if decision == "Error" or confidence < 75:
        print(f"\n>>> [二次路由] 两次结果不一致或置信度不足 ({confidence})，转入人工审查...")
        return "human_review"
    else:
        print(f"\n>>> [二次路由] 二次验证结论: {decision} (置信度 {confidence})，直接输出。")
        return "end"

# --- 构建 LangGraph 工作流 ---
workflow = StateGraph(GraphState)

workflow.add_node("trap_check", trap_check_node)
workflow.add_node("solve", solve_node)
workflow.add_node("code_executor", code_executor_node)
workflow.add_node("judge", judge_node)
workflow.add_node("retry_solve", retry_solve_node)
workflow.add_node("retry_executor", retry_executor_node)
workflow.add_node("retry_judge", retry_judge_node)
workflow.add_node("human_review", human_review_node)

# 起点 → 可解性检测
workflow.add_edge(START, "trap_check")

# 可解性检测后：不可解 → 结束，可解 → 解题
workflow.add_conditional_edges(
    "trap_check",
    route_after_trap_check,
    {
        "end": END,
        "solve": "solve"
    }
)

# 解题 → 执行 → 裁判
workflow.add_edge("solve", "code_executor")
workflow.add_edge("code_executor", "judge")

# 裁判后三路分支：Match → 结束，Mismatch → 二次验证，低置信/Error → HITL
workflow.add_conditional_edges(
    "judge",
    route_after_judge,
    {
        "end": END,
        "retry_solve": "retry_solve",
        "human_review": "human_review"
    }
)

# 二次验证链路：重新解题 → 执行 → 二次裁判
workflow.add_edge("retry_solve", "retry_executor")
workflow.add_edge("retry_executor", "retry_judge")

# 二次裁判后：结果一致 → 结束，不一致 → HITL
workflow.add_conditional_edges(
    "retry_judge",
    route_after_retry_judge,
    {
        "end": END,
        "human_review": "human_review"
    }
)

# 人工审查 → 结束
workflow.add_edge("human_review", END)

import uuid

# 借助 Checkpointer 实现图执行中断
memory = MemorySaver()

# 编译 Graph，指定在 human_review 节点前发生中断 (interrupt_before)
graph_app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review"]
)

if __name__ == "__main__":
    print("欢迎使用 LangGraph 工作流控制台！")
    print("请发送包含问题和参考答案的消息，使用 `|` 分隔。")
    print("例如：`在 $7$ 维单位超正方体（所有坐标均在 $[0, 1]$ 之间）中，随机均匀地取出 $5$ 个点。如果点 $A$ 的所有坐标轴数值都大于点 $B$，则称 $A$ 支配 $B$。求这 $5$ 个点中，没有任何一个点被其他点“支配”的概率。 | 0.87`")
    print("输入 'q' 退出。\n")
    
    while True:
        user_input = input("输入题目|答案: ")
        if user_input.lower() in ('q', 'quit', 'exit'):
            break
            
        if "|" not in user_input:
            print("出错了，请使用 `|` 分隔问题和参考答案。\n")
            continue
            
        parts = user_input.split("|")
        question = parts[0].strip()
        truth = parts[1].strip()
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "question_id": f"q_{thread_id[:4]}",
            "question_context": question,
            "ground_truth": truth
        }
        
        print("\n运行工作流中...\n")
        
        # 运行到断点或者结束
        for _ in graph_app.stream(initial_state, config=config, stream_mode="values"):
            pass
            
        state_info = graph_app.get_state(config)
        state_str = json.dumps(state_info.values, ensure_ascii=False, indent=2)
        print(f"**当前图状态 (Graph State):**\n{state_str}\n")
        
        needs_hitl = state_info.next and "human_review" in state_info.next
        
        if needs_hitl:
            print("⚠️ **触发人工审批 (HITL)**")
            print("系统置信度低，需要人工介入判定：")
            print("1. 认同结论 (Pass)")
            print("2. 推翻结论 (Fail)")
            print("3. 继续 (不修改)")
            
            choice = input("请选择 (1/2/3): ")
            decision = None
            if choice == "1":
                decision = "Manual_Confirmed_Match"
            elif choice == "2":
                decision = "Manual_Overruled_Mismatch"
            
            if decision:
                graph_app.update_state(config, {"final_decision": decision})
                
            print("\n恢复执行中...\n")
            for _ in graph_app.stream(None, config=config, stream_mode="values"):
                pass
                
            state_info = graph_app.get_state(config)
            state_str = json.dumps(state_info.values, ensure_ascii=False, indent=2)
            print(f"**最终图状态:**\n{state_str}\n")
        
        print("-" * 50 + "\n")