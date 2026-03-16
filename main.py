from typing import TypedDict, Any, Literal
import os
import json
import re
import docker
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
client = OpenAI()

class ClassificationResult(BaseModel):
    problem_type: Literal["几何", "代数", "概率", "数论"]
    hierarchy: Literal["初中", "高中", "本科", "硕士及以上"]
    difficulty: Literal["基础", "进阶", "竞赛"]

class JudgeResult(BaseModel):
    confidence: int
    decision: Literal["Match", "Mismatch", "Error"]

class GraphState(TypedDict):
    question_id: str
    question_context: str
    ground_truth: str
    problem_type: str
    trap_analysis: bool
    generated_code: str
    execution_output: str
    confidence_score: float
    final_decision: str
    hierarchy: str
    difficulty: str


def type_classifier_node(state: GraphState):
    """Node 1: 分类器"""
    question = state.get("question_context", "")
    prompt = os.getenv("TYPE_CLASSIFIER_PROMPT", "You are an expert math problem classifier...")
    
    try:
        response = client.beta.chat.completions.parse(
            model="gemini-3-flash-preview",
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
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # 如果调用失败时的默认 fallback，保证流程继续
        return {
            "problem_type": "代数", 
            "hierarchy": "高中", 
            "difficulty": "基础"
        }

def code_generator_node(state: GraphState):
    """Node 2: 代码生成器"""
    question = state.get("question_context", "")
    problem_type = state.get("problem_type", "代数")
    
    # 根据题目类型加载对应的 prompt
    prompt_env_map = {
        "几何": "CODE_GEN_PROMPT_几何",
        "代数": "CODE_GEN_PROMPT_代数",
        "概率": "CODE_GEN_PROMPT_概率",
        "数论": "CODE_GEN_PROMPT_数论"
    }
    env_key = prompt_env_map.get(problem_type, "CODE_GEN_PROMPT_代数")
    system_prompt = os.getenv(env_key, "You are a helpful coding assistant.")
    
    try:
        response = client.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please solve the following problem:\n\n{question}"}
            ],
            temperature=1.0
        )
        content = response.choices[0].message.content
        
        # 使用正则提取 Markdown 格式中的 python 代码块
        match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            # 如果没有找到代码块，直接返回内容或者做兜底处理
            generated_code = content
            
        return {"generated_code": generated_code}
    except Exception as e:
        print(f"Error in code generator: {e}")
        return {"generated_code": "print('Error generating code')"}


def trap_classifier_node(state: GraphState):
    """Node 3: 陷阱分类器"""
    question = state.get("question_context", "")
    problem_type = state.get("problem_type", "")
    
    # 针对几何题目使用独立的Prompt，其余题目使用通用的Prompt
    if problem_type == "几何":
        system_prompt = os.getenv("TRAP_CLASSIFIER_PROMPT_几何", "You are an expert at identifying logical traps in geometry problems. You must include either '可解' or '不可解' in your response.")
    else:
        system_prompt = os.getenv("TRAP_CLASSIFIER_PROMPT", "You are an expert at identifying logical traps in math problems. You must include either '可解' or '不可解' in your response.")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-3-flash-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the following problem for logical traps or missing conditions:\n\n{question}\n\nRespond with a comprehensive trap analysis. Note: Your response MUST contain the exact word '不可解' if it has traps/missing conditions, or '可解' if it is solvable."}
                ],
                temperature=1.0
            )
            content = response.choices[0].message.content
            
            if "不可解" in content:
                return {"trap_analysis": True}
            elif "可解" in content:
                return {"trap_analysis": False}
            else:
                print(f"--- [Trap Classifier] 未检测到 '可解' 或 '不可解'，正在打回重新生成... ({attempt+1}/{max_retries}) ---")
                continue
        except Exception as e:
            print(f"Error in trap classifier: {e}")
            return {"trap_analysis": False}
            
    # 如果达到最大重试次数仍然不符合格式，默认当做可解题（无陷阱）以保证流转
    print("--- [Trap Classifier] 超过最大重试次数，默认标记为可解(False)。 ---")
    return {"trap_analysis": False}

def parallel_sync_node(state: GraphState):
    """虚拟同步节点: 汇聚并行的 Node 2 和 Node 3 结果"""
    # 状态汇聚会自动处理并更新对应的字段，这里无需做额外修改直接返回空字典或者是需要整理的其他字段
    return {}

def code_executor_node(state: GraphState):
    """Node 4: 安全代码沙盒执行器"""
    generated_code = state.get("generated_code", "")
    if not generated_code.strip():
        return {"execution_output": "Error: No code to execute."}

    execution_output = ""
    try:
        docker_client = docker.from_env()
        # 启动沙盒容器（分离模式，避免直接阻塞主进程导致主Python卡死）
        # 这里限制内存为 1g，并且不映射任何本地卷，以防恶意操作系统级破坏
        container = docker_client.containers.run(
            image="python:3.10-slim",
            command=["python", "-c", generated_code],
            detach=True,
            mem_limit="1g",
            network_disabled=True # 断网策略提升安全
        )
        
        try:
            # 阻塞等待最多 60 秒
            result = container.wait(timeout=60)
            
            # 分离获取 stdout 和 stderr
            stdout_logs = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr_logs = container.logs(stdout=False, stderr=True).decode("utf-8")
            
            output_parts = []
            if stdout_logs:
                output_parts.append(f"----- STDOUT -----\n{stdout_logs}")
            if stderr_logs:
                output_parts.append(f"----- STDERR -----\n{stderr_logs}")
            if result.get("StatusCode", 0) != 0:
                output_parts.append(f"Exit Code: {result.get('StatusCode')}")
                
            execution_output = "\n".join(output_parts) if output_parts else "Success (No Output)"
            
        except requests.exceptions.ReadTimeout:
            # 如果超时，则强制销毁进程并记录报错信息
            container.kill()
            execution_output = "Error: Code execution exceeded the 60 seconds timeout."
            
        finally:
            # 清理容器释放资源
            container.remove(force=True)
            
    except Exception as e:
        print(f"Docker API Error: {e}")
        execution_output = f"Execution setup failed: {str(e)}"

    return {"execution_output": execution_output}

def judge_node(state: GraphState):
    """Node 5: 裁判 (The Judge)"""
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
    
    try:
        response = client.beta.chat.completions.parse(
            model="gemini-3-flash-preview",
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
            "final_decision": result.decision
        }
    except Exception as e:
        print(f"Error calling OpenAI API in judge: {e}")
        return {
            "confidence_score": 0.0,
            "final_decision": "Error"
        }

def human_review_node(state: GraphState):
    """Node 6: 人工审查节点 (HITL)"""
    # 实际运行中，执行到此节点前会被打断。当恢复执行时，会运行此节点。
    print("--- [HITL] 人工审查节点触发，处理并向下流转 ---")
    return {}

def route_after_judge(state: GraphState) -> str:
    """条件路由逻辑"""
    trap_analysis = state.get("trap_analysis", False)
    confidence = state.get("confidence_score", 0.0)

    # 1. 发现陷阱 -> 废题
    if trap_analysis is True:
        print("\n>>> [路由] 发现逻辑陷阱，直接废弃结论。")
        return "end"
    
    # 2. 裁判置信度低 -> 进人工打断
    elif confidence < 75:
        print(f"\n>>> [路由] 置信度为 {confidence} < 75，转入人工审查 (HITL)...")
        return "human_review"
        
    # 3. 否则 -> 直通终点
    else:
        print(f"\n>>> [路由] 置信度为 {confidence} >= 75，直接输出终局结论。")
        return "end"

# --- 构建 LangGraph 工作流 ---
workflow = StateGraph(GraphState)

workflow.add_node("type_classifier", type_classifier_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("trap_classifier", trap_classifier_node)
workflow.add_node("parallel_sync", parallel_sync_node)
workflow.add_node("code_executor", code_executor_node)
workflow.add_node("judge", judge_node)
workflow.add_node("human_review", human_review_node)

# 从起点连向类型分类器
workflow.add_edge(START, "type_classifier")

# 节点 1 将图分叉流向 Node 2 (代码生成器) 以及 Node 3 (陷阱分类器)
workflow.add_edge("type_classifier", "code_generator")
workflow.add_edge("type_classifier", "trap_classifier")

# Node 2 和 Node 3 汇聚走到同步节点
workflow.add_edge("code_generator", "parallel_sync")
workflow.add_edge("trap_classifier", "parallel_sync")

# 同步节点流向 Node 4 代码执行器
workflow.add_edge("parallel_sync", "code_executor")

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
    print("例如：`已知 a+b=3，a-b=1，求 a*b。 | 2`")
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
