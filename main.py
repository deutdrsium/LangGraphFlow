from typing import TypedDict, Any, Literal
import os
import json
import re
import time
import subprocess
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()
client = OpenAI()

# 全局存储流式输出用于前端展示
streaming_store = {}


class ClassificationResult(BaseModel):
    problem_type: Literal["几何", "代数", "概率", "数论"]
    hierarchy: Literal["初中", "高中", "本科", "硕士及以上"]
    difficulty: Literal["基础", "进阶", "竞赛"]

class JudgeResult(BaseModel):
    confidence: int
    decision: Literal["Match", "Mismatch", "Error"]
    verified_ans: str

class GraphState(TypedDict):
    question_id: str
    question_context: str
    ground_truth: str
    problem_type: str
    trap_analysis: bool
    trap_reason: str
    generated_code: str
    execution_output: str
    confidence_score: float
    final_decision: str
    verified_ans: str
    hierarchy: str
    difficulty: str


def type_classifier_node(state: GraphState):
    """Node 1: 分类器"""
    print("\n---> [Node: type_classifier] 开始执行...", flush=True)
    question = state.get("question_context", "")
    prompt = os.getenv("TYPE_CLASSIFIER_PROMPT", "You are an expert math problem classifier...")
    support_structured = os.getenv("SUPPORT_STRUCTURED_OUTPUT", "True").lower() == "true"
    
    try:
        if support_structured:
            response = client.beta.chat.completions.parse(
                model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Please classify the following question:\n\n{question}"}
                ],
                response_format=ClassificationResult,
                temperature=1.0
            )
            result = response.choices[0].message.parsed
            return {
                "problem_type": result.problem_type,
                "hierarchy": result.hierarchy,
                "difficulty": result.difficulty
            }
        else:
            prompt_with_instructions = prompt + "\n\nPlease return ONLY a JSON object string exactly matching this schema: {\"problem_type\": \"几何|代数|概率|数论\", \"hierarchy\": \"初中|高中|本科|硕士及以上\", \"difficulty\": \"基础|进阶|竞赛\"}. Do not use code blocks."
            response = client.chat.completions.create(
                model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt_with_instructions},
                    {"role": "user", "content": f"Please classify the following question:\n\n{question}"}
                ],
                temperature=1.0
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result_dict = json.loads(content)
            return {
                "problem_type": result_dict.get("problem_type", "代数"),
                "hierarchy": result_dict.get("hierarchy", "高中"),
                "difficulty": result_dict.get("difficulty", "基础")
            }
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # 如果调用失败时的默认 fallback，保证流程继续
        return {
            "problem_type": "代数", 
            "hierarchy": "高中", 
            "difficulty": "基础"
        }

def analyze_and_solve_node(state: GraphState, config: RunnableConfig = None):
    """Node 2: 陷阱分析 + 代码生成 (合并节点，使用强模型)"""
    print("\n---> [Node: analyze_and_solve] 开始执行...", flush=True)
    question = state.get("question_context", "")
    problem_type = state.get("problem_type", "代数")
    thread_id = config.get("configurable", {}).get("thread_id", None) if config else None

    # 根据题目类型加载对应的合并 prompt
    prompt_env_map = {
        "几何": "ANALYZE_AND_SOLVE_PROMPT_几何",
        "代数": "ANALYZE_AND_SOLVE_PROMPT_代数",
        "概率": "ANALYZE_AND_SOLVE_PROMPT_概率",
        "数论": "ANALYZE_AND_SOLVE_PROMPT_数论"
    }
    env_key = prompt_env_map.get(problem_type, "ANALYZE_AND_SOLVE_PROMPT_代数")
    system_prompt = os.getenv(env_key, "You are a helpful math assistant.")

    attempt = 0
    error_codes = []

    while True:
        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_PRO", "gemini-3.1-pro-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析并求解以下题目：\n\n{question}"}
                ],
                temperature=1.0,
                stream=True,
                timeout=60.0
            )

            display_content = ""  # 用于前端实时展示，包含 CoT 和正文
            final_content = ""    # 仅包含正文，用于最终解析

            print("\n--- [Analyze & Solve] 开始流式输出 ---\n", end="")
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    delta_dict = delta.model_dump()

                    # 1. 获取并拼接思维链内容 (CoT)
                    reasoning = delta_dict.get("reasoning_content")
                    if reasoning:
                        display_content += reasoning
                        print(reasoning, end="", flush=True)

                    # 2. 获取并拼接最终正文内容
                    content = delta.content
                    if content:
                        display_content += content
                        final_content += content
                        print(content, end="", flush=True)

                    # 将包含 CoT 的完整记录更新到 WebUI 存储中
                    if thread_id:
                        streaming_store[thread_id] = display_content

            print("\n--- [Analyze & Solve] 流式输出结束 ---\n")

            # 判断模型是否拒绝答题（检测到陷阱）
            if "[TRAP_DETECTED]" in final_content:
                trap_text = final_content.split("[TRAP_DETECTED]", 1)[1].strip()
                # 取第一行作为 trap_reason，截断到100字
                trap_reason = trap_text.split("\n")[0].strip()[:100]
                if not trap_reason:
                    trap_reason = "模型检测到逻辑陷阱但未给出原因"

                print(f"\n>>> [陷阱检测] 发现逻辑陷阱: {trap_reason}")
                return {
                    "trap_analysis": True,
                    "trap_reason": trap_reason,
                    "generated_code": "pass"
                }
            else:
                # 正常代码生成，提取 python 代码块
                match = re.search(r'```python\n(.*?)\n```', final_content, re.DOTALL)
                if match:
                    generated_code = match.group(1).strip()
                else:
                    generated_code = final_content

                return {
                    "trap_analysis": False,
                    "trap_reason": "pass",
                    "generated_code": generated_code
                }

        except Exception as e:
            attempt += 1
            is_429 = (hasattr(e, 'status_code') and e.status_code == 429) or '429' in str(e)
            error_codes.append(429 if is_429 else 0)
            print(f"Error in analyze_and_solve (attempt {attempt}): {e}", flush=True)

            if attempt >= 3:
                # 检查最近3次是否全部为429
                if all(code == 429 for code in error_codes[-3:]):
                    print(f"连续3次触发429限流，等待15s后继续重试...", flush=True)
                    time.sleep(15)
                    attempt = 0
                    error_codes.clear()
                    continue
                else:
                    raise Exception(f"analyze_and_solve 连续失败3次 (非全部429限流)，阻塞: {e}")

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
    
    prompt = os.getenv("JUDGE_PROMPT", "You are an expert mathematical judge. Compare the code execution output with the ground truth for the given question.")
    
    # 组装完整的判断内容
    user_content = f"""
[Original Question]
{question}

[Ground Truth]
{ground_truth}

[Sandbox Execution Output]
{execution_output}
"""
    support_structured = os.getenv("SUPPORT_STRUCTURED_OUTPUT", "True").lower() == "true"
    
    try:
        if support_structured:
            response = client.beta.chat.completions.parse(
                model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=JudgeResult,
                temperature=1.0
            )
            result = response.choices[0].message.parsed
            return {
                "confidence_score": float(result.confidence),
                "final_decision": result.decision,
                "verified_ans": result.verified_ans
            }
        else:
            prompt_with_instructions = prompt + "\n\nPlease return ONLY a JSON object string exactly matching this schema: {\"confidence\": <int>, \"decision\": \"Match\" or \"Mismatch\" or \"Error\", \"verified_ans\": <string>}. Use 'pass' for verified_ans if decision is Match. Do not use code blocks."
            response = client.chat.completions.create(
                model=os.getenv("MODEL_FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt_with_instructions},
                    {"role": "user", "content": user_content}
                ],
                temperature=1.0
            )
            content = response.choices[0].message.content.strip()
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

def human_review_node(state: GraphState):
    """Node 6: 人工审查节点 (HITL)"""
    # 实际运行中，执行到此节点前会被打断。当恢复执行时，会运行此节点。
    print("--- [HITL] 人工审查节点触发，处理并向下流转 ---")
    return {}

def route_after_analyze(state: GraphState) -> str:
    """如果发现陷阱，不再执行代码，直接去结尾"""
    if state.get("trap_analysis", False) is True:
        print(f"\n>>> [路由] 发现逻辑陷阱: {state.get('trap_reason')}\n>>> 终止其余节点，直接输出。")
        return "end"
    return "code_executor"

def route_after_judge(state: GraphState) -> str:
    """条件路由逻辑"""
    confidence = state.get("confidence_score", 0.0)

    # 1. 裁判置信度低 -> 进人工打断
    if confidence < 75:
        print(f"\n>>> [路由] 置信度为 {confidence} < 75，转入人工审查 (HITL)...")
        return "human_review"
        
    # 2. 否则 -> 直通终点
    else:
        print(f"\n>>> [路由] 置信度为 {confidence} >= 75，直接输出终局结论。")
        return "end"

# --- 构建 LangGraph 工作流 ---
workflow = StateGraph(GraphState)

workflow.add_node("type_classifier", type_classifier_node)
workflow.add_node("analyze_and_solve", analyze_and_solve_node)
workflow.add_node("code_executor", code_executor_node)
workflow.add_node("judge", judge_node)
workflow.add_node("human_review", human_review_node)

# 从起点连向类型分类器
workflow.add_edge(START, "type_classifier")

# 类型分类器流向合并节点（陷阱分析 + 代码生成）
workflow.add_edge("type_classifier", "analyze_and_solve")

# 根据合并节点的结果，如果有陷阱则直接终止到 END；否则正常流转到代码执行器
workflow.add_conditional_edges(
    "analyze_and_solve",
    route_after_analyze,
    {
        "end": END,
        "code_executor": "code_executor"
    }
)

# 代码执行器流向裁判节点
workflow.add_edge("code_executor", "judge")

# --------- 新增的条件边与 HITL ---------
# 根据 Node 5 裁判的结果通过 route_after_judge 函数发往终点或人工审核
workflow.add_conditional_edges(
    "judge",
    route_after_judge,
    {
        "end": END,
        "human_review": "human_review"
    }
)

# 人工节点流向终点
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