from typing import TypedDict, Any, Literal
import os
import json
import re
import time
import docker
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from rate_limiter import pro_limiter, flash_limiter

load_dotenv()
client = OpenAI()

# 全局存储流式输出用于前端展示
streaming_store = {}

BUILTIN_HELPER_MODULES = {"wolfram_eval"}
PYTHON_FENCE_LANGS = {"python", "py", "python3", "python2", "ipython"}
GENERIC_FENCE_LANGS = {"", "code", "text", "txt"}


def reasoning_effort_kwargs(model_tier: str) -> dict:
    effort = os.getenv(f"MODEL_{model_tier}_REASONING_EFFORT", "").strip().lower()
    if not effort or effort in {"none", "off", "false", "0"}:
        return {}
    return {"reasoning_effort": effort}


def code_extraction_max_retries() -> int:
    try:
        retries = int(os.getenv("CODE_EXTRACTION_MAX_RETRIES", "1"))
    except ValueError:
        retries = 1
    return max(0, min(retries, 3))


def code_retry_user_prompt(question: str, previous_output: str) -> str:
    previous_output = (previous_output or "").strip()
    if len(previous_output) > 6000:
        previous_output = previous_output[-6000:]
    return (
        "The previous model response did not contain any executable Python code. "
        "This is a recoverable generation failure.\n\n"
        "Return exactly one of the following:\n"
        "1. If the problem is invalid or impossible, output [TRAP_DETECTED] followed by a short reason.\n"
        "2. Otherwise, output ONLY executable Python code inside a ```python fenced block. "
        "The code must print the final answer to STDOUT. Do not include analysis, reasoning, markdown prose, "
        "or any text outside the code fence.\n\n"
        f"[Original Question]\n{question}\n\n"
        f"[Previous Non-Executable Response]\n{previous_output}"
    )


def anthropic_messages_enabled() -> bool:
    protocol = os.getenv("LLM_API_PROTOCOL", os.getenv("LLM_COMPAT_MODE", "openai")).strip().lower()
    return protocol in {"anthropic", "anthropic_messages", "anthropic-messages", "anthropic_messages_compat"}


def llm_model_name(model_tier: str, default_model: str) -> str:
    return os.getenv(f"MODEL_{model_tier}", default_model).strip()


def anthropic_api_key() -> str:
    return (
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def anthropic_auth_header() -> str:
    key = anthropic_api_key()
    if key.lower().startswith("bearer "):
        return key
    return f"Bearer {key}"


def anthropic_custom_headers() -> dict:
    return {
        "Authorization": anthropic_auth_header(),
        "User-Agent": os.getenv("ANTHROPIC_USER_AGENT", "claude-cli/2.0.76 (external, cli)"),
    }


def create_chat_completion(model_tier: str, default_model: str, **kwargs):
    if not anthropic_messages_enabled():
        return client.chat.completions.create(
            model=llm_model_name(model_tier, default_model),
            **kwargs,
        )

    try:
        import litellm
    except ImportError as e:
        raise RuntimeError("LLM_API_PROTOCOL=anthropic_messages requires `pip install litellm`.") from e

    base_url = (
        os.getenv("ANTHROPIC_BASE_URL", "").strip()
        or os.getenv("ANTHROPIC_API_BASE", "").strip()
    )
    kwargs.pop("reasoning_effort", None)
    call_kwargs = {
        "model": llm_model_name(model_tier, default_model),
        "api_key": anthropic_api_key(),
        "extra_headers": anthropic_custom_headers(),
        "drop_params": True,
        **kwargs,
    }
    if base_url:
        call_kwargs["api_base"] = base_url
        call_kwargs["base_url"] = base_url
    return litellm.completion(**call_kwargs)


def _clean_code_candidate(code: str) -> str:
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    code = re.sub(r"^\s*(?:python|py|python3)\s*\n", "", code, flags=re.IGNORECASE)
    return code.strip()


def _compile_score(code: str) -> int:
    if not code.strip():
        return -100
    try:
        compile(code, "<generated_code>", "exec")
        return 50
    except SyntaxError:
        return -20


def _is_compilable_python(code: str) -> bool:
    if not code.strip():
        return False
    try:
        compile(code, "<generated_code>", "exec")
        return True
    except SyntaxError:
        return False


CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"#|import |from |def |class |for |while |if |elif |else:|try:|except |finally:|"
    r"with |return\b|raise\b|assert\b|print\s*\(|"
    r"[\w.]+\s*=|[\w.]+\s*\(|[\]\)\}]"
    r")"
)


def _has_python_execution_signal(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return False
    signal_patterns = [
        r"^\s*(?:import|from)\s+\w+",
        r"^\s*(?:def|class|for|while|if|try|with)\b",
        r"\bprint\s*\(",
        r"\bwolfram_eval\s*\(",
        r"\bsys\.stdout\.write\s*\(",
        r"^\s*\w[\w.]*\s*=",
    ]
    return any(re.search(pattern, stripped, re.MULTILINE) for pattern in signal_patterns)


def _trim_to_compilable_python(code: str) -> str:
    cleaned = _clean_code_candidate(code)
    if _is_compilable_python(cleaned) and _has_python_execution_signal(cleaned):
        return cleaned

    lines = cleaned.splitlines()
    code_indices = [i for i, line in enumerate(lines) if CODE_LINE_RE.match(line)]
    if not code_indices:
        return ""

    trimmed = "\n".join(lines[code_indices[0] : code_indices[-1] + 1]).strip()
    if _is_compilable_python(trimmed) and _has_python_execution_signal(trimmed):
        return trimmed

    # Last-resort salvage: find the largest compilable contiguous code-looking block.
    best = ""
    starts = code_indices[:]
    for start in starts:
        for end in range(len(lines), start, -1):
            chunk = "\n".join(lines[start:end]).strip()
            if len(chunk) <= len(best):
                continue
            if _is_compilable_python(chunk) and _has_python_execution_signal(chunk):
                best = chunk
                break
    return best


def _python_signal_score(code: str) -> int:
    stripped = code.strip()
    if not stripped:
        return -100

    score = 0
    strong_patterns = [
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bprint\s*\(",
        r"\bwolfram_eval\s*\(",
        r"\b(?:sympy|numpy|scipy|math|itertools|fractions|decimal)\b",
        r"\b(?:sp|np)\.",
        r"^\s*(?:for|while|if|with|try|except|return|raise|assert)\b",
        r"^\s*\w[\w.]*\s*=",
    ]
    for pattern in strong_patterns:
        if re.search(pattern, stripped, re.MULTILINE):
            score += 8

    prose_markers = [
        "```",
        "Here is",
        "The code",
        "I will",
        "We need",
        "Let's",
        "下面",
        "代码如下",
    ]
    for marker in prose_markers:
        if marker in stripped:
            score -= 8

    code_like_lines = 0
    prose_like_lines = 0
    for line in stripped.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(
            r"^(?:#|import |from |def |class |for |while |if |elif |else:|try:|except |finally:|with |return |raise |assert |print\(|[\w.]+\s*=)",
            s,
        ):
            code_like_lines += 1
        elif len(s.split()) >= 8 and not any(token in s for token in ("=", "(", ")", "[", "]", "{", "}")):
            prose_like_lines += 1

    score += code_like_lines * 3
    score -= prose_like_lines * 5
    score += _compile_score(stripped)
    return score


def _fenced_code_candidates(text: str):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[^\n`]*)\n(?P<code>.*?)(?:\n(?P=fence)[ \t]*(?=\n|$)|$)",
        re.DOTALL,
    )
    for index, match in enumerate(pattern.finditer(normalized)):
        lang = (match.group("lang") or "").strip().lower().split()
        lang = lang[0] if lang else ""
        code = _clean_code_candidate(match.group("code"))
        if not code:
            continue
        code = _trim_to_compilable_python(code)
        if not code:
            continue
        if lang in PYTHON_FENCE_LANGS:
            priority = 100
        elif lang in GENERIC_FENCE_LANGS:
            priority = 35
        else:
            priority = -15
        yield priority + _python_signal_score(code), index, code


def _tagged_code_candidates(text: str):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"<(?P<tag>python|py|code)>(?P<code>.*?)</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    )
    for index, match in enumerate(pattern.finditer(normalized)):
        code = _trim_to_compilable_python(match.group("code"))
        if code:
            yield 70 + _python_signal_score(code), index, code


def _line_block_candidates(text: str):
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    code_line = re.compile(
        r"^\s*(?:#|import |from |def |class |for |while |if |elif |else:|try:|except |finally:|with |return |raise |assert |print\(|[\w.]+\s*=|[\w.]+\s*\(|\])"
    )

    blocks = []
    current = []
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped and current:
            current.append(line)
            continue
        if code_line.match(line):
            if not current:
                start = index
            current.append(line)
            continue
        if current:
            blocks.append((start, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, "\n".join(current)))

    for index, block in blocks:
        code = _clean_code_candidate(block)
        if code and _is_compilable_python(code) and _has_python_execution_signal(code):
            yield 20 + _python_signal_score(code), index, code


def extract_python_code(model_output: str) -> str:
    """Extract executable Python from an LLM answer with tolerant fallbacks."""
    text = (model_output or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    candidates = []
    candidates.extend(_fenced_code_candidates(text))
    candidates.extend(_tagged_code_candidates(text))
    candidates.extend(_line_block_candidates(text))

    raw = _clean_code_candidate(text)
    if raw and _is_compilable_python(raw) and _has_python_execution_signal(raw):
        raw_score = _python_signal_score(raw)
        candidates.append((raw_score, 10_000, raw))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return candidates[0][2]

# Docker 客户端单例（设置较长超时以应对 Windows Named Pipe 偶发延迟）
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
        if flash_limiter:
            print("[Rate Limit] Waiting for MODEL_FLASH slot (type_classifier)...", flush=True)
            flash_limiter.acquire()
        if support_structured and not anthropic_messages_enabled():
            response = client.beta.chat.completions.parse(
                model=llm_model_name("FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Please classify the following question:\n\n{question}"}
                ],
                response_format=ClassificationResult,
                temperature=1.0,
                **reasoning_effort_kwargs("FLASH"),
            )
            result = response.choices[0].message.parsed
            return {
                "problem_type": result.problem_type,
                "hierarchy": result.hierarchy,
                "difficulty": result.difficulty
            }
        else:
            prompt_with_instructions = prompt + "\n\nPlease return ONLY a JSON object string exactly matching this schema: {\"problem_type\": \"几何|代数|概率|数论\", \"hierarchy\": \"初中|高中|本科|硕士及以上\", \"difficulty\": \"基础|进阶|竞赛\"}. Do not use code blocks."
            response = create_chat_completion(
                "FLASH",
                "gemini-3-flash-preview",
                messages=[
                    {"role": "system", "content": prompt_with_instructions},
                    {"role": "user", "content": f"Please classify the following question:\n\n{question}"}
                ],
                temperature=1.0,
                **reasoning_effort_kwargs("FLASH"),
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
    extraction_attempt = 0
    max_extraction_retries = code_extraction_max_retries()

    while True:
        try:
            if pro_limiter:
                print("[Rate Limit] Waiting for MODEL_PRO slot...", flush=True)
                pro_limiter.acquire()
            user_prompt = (
                f"请分析并求解以下题目：\n\n{question}"
                if extraction_attempt == 0
                else code_retry_user_prompt(question, final_content)
            )
            response = create_chat_completion(
                "PRO",
                "gemini-3.1-pro-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                stream=True,
                timeout=60.0,
                **reasoning_effort_kwargs("PRO"),
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
                trap_reason = final_content.split("[TRAP_DETECTED]", 1)[1].strip()
                if not trap_reason:
                    trap_reason = "模型检测到逻辑陷阱但未给出原因"

                print(f"\n>>> [陷阱检测] 发现逻辑陷阱: {trap_reason}")
                return {
                    "trap_analysis": True,
                    "trap_reason": trap_reason,
                    "generated_code": "pass"
                }
            else:
                generated_code = extract_python_code(final_content)
                if not generated_code.strip():
                    if extraction_attempt < max_extraction_retries:
                        extraction_attempt += 1
                        print(
                            f"No executable Python code extracted; retrying code generation "
                            f"({extraction_attempt}/{max_extraction_retries})...",
                            flush=True,
                        )
                        continue
                    generated_code = "raise RuntimeError('No executable Python code was extracted from the model output after retry.')"

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
    """Node 4: 安全代码沙盒执行器"""
    print("\n---> [Node: code_executor] 开始执行...", flush=True)
    generated_code = state.get("generated_code", "")
    if not generated_code.strip():
        return {"execution_output": "Error: No code to execute."}

    execution_output = ""
    try:
        docker_client = _get_docker_client()
        base_image = "math_sandbox:latest"
        
        # 1. 检查并构建包含常见库的基础镜像
        try:
            docker_client.images.get(base_image)
        except docker.errors.ImageNotFound:
            print("未找到基础沙盒镜像，正在构建包含常用数学库的 Docker 镜像 (可能需要几分钟)...")
            dockerfile = "FROM python:3.10-slim\nRUN pip install numpy sympy z3-solver networkx matplotlib scipy"
            from io import BytesIO
            docker_client.images.build(fileobj=BytesIO(dockerfile.encode('utf-8')), tag=base_image, rm=True)
            print("沙盒镜像构建完成。")

        current_image = base_image
        max_retries = 3

        for attempt in range(max_retries):
            # 启动沙盒容器（分离模式，避免直接阻塞主进程导致主Python卡死）
            # 这里限制内存为 1g，并且不映射任何本地卷，以防恶意操作系统级破坏
            container = docker_client.containers.run(
                image=current_image,
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
                
                # 检查是否因为缺少库导致 ModuleNotFoundError
                if "ModuleNotFoundError: No module named" in stderr_logs:
                    import re
                    match = re.search(r"No module named '([^']+)'", stderr_logs)
                    if match and attempt < max_retries - 1:
                        missing_module = match.group(1)
                        if missing_module in BUILTIN_HELPER_MODULES:
                            break
                        print(f"检测到缺失库: {missing_module}，准备重新构建镜像并安装...")
                        new_image_tag = f"math_sandbox_with_{missing_module}:latest"
                        dockerfile = f"FROM {current_image}\nRUN pip install {missing_module}"
                        from io import BytesIO
                        # 构建包含缺失库的新镜像（此处会自动联网拉取依赖，但随后代码运行仍断网）
                        docker_client.images.build(fileobj=BytesIO(dockerfile.encode('utf-8')), tag=new_image_tag, rm=True)
                        current_image = new_image_tag
                        container.remove(force=True)
                        continue  # 重试执行

                # 成功或其他错误，跳出重试循环
                break
                
            except requests.exceptions.ReadTimeout:
                # 如果超时，则强制销毁进程并记录报错信息
                container.kill()
                execution_output = "Error: Code execution exceeded the 60 seconds timeout."
                break
                
            finally:
                # 清理容器释放资源
                try:
                    container.remove(force=True)
                except Exception:
                    pass
                
    except Exception as e:
        print(f"Docker API Error: {e}")
        execution_output = f"Execution setup failed: {str(e)}"

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
        if flash_limiter:
            print("[Rate Limit] Waiting for MODEL_FLASH slot (judge)...", flush=True)
            flash_limiter.acquire()
        if support_structured and not anthropic_messages_enabled():
            response = client.beta.chat.completions.parse(
                model=llm_model_name("FLASH", "gemini-3-flash-preview"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=JudgeResult,
                temperature=1.0,
                **reasoning_effort_kwargs("FLASH"),
            )
            result = response.choices[0].message.parsed
            return {
                "confidence_score": float(result.confidence),
                "final_decision": result.decision,
                "verified_ans": result.verified_ans
            }
        else:
            prompt_with_instructions = prompt + "\n\nPlease return ONLY a JSON object string exactly matching this schema: {\"confidence\": <int>, \"decision\": \"Match\" or \"Mismatch\" or \"Error\", \"verified_ans\": <string>}. Use 'pass' for verified_ans if decision is Match. Do not use code blocks."
            response = create_chat_completion(
                "FLASH",
                "gemini-3-flash-preview",
                messages=[
                    {"role": "system", "content": prompt_with_instructions},
                    {"role": "user", "content": user_content}
                ],
                temperature=1.0,
                **reasoning_effort_kwargs("FLASH"),
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
