import ast
import os
import unittest
from pathlib import Path


def _load_extract_python_code():
    source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    needed = {
        "PYTHON_FENCE_LANGS",
        "GENERIC_FENCE_LANGS",
        "CODE_LINE_RE",
        "_clean_code_candidate",
        "_compile_score",
        "_is_compilable_python",
        "_has_python_execution_signal",
        "_trim_to_compilable_python",
        "_python_signal_score",
        "_fenced_code_candidates",
        "_tagged_code_candidates",
        "_line_block_candidates",
        "extract_python_code",
        "code_extraction_max_retries",
        "code_retry_user_prompt",
        "anthropic_messages_enabled",
        "llm_model_name",
        "anthropic_api_key",
        "anthropic_auth_header",
        "anthropic_custom_headers",
    }
    selected = [
        ast.Import(names=[ast.alias(name="os")]),
        ast.Import(names=[ast.alias(name="re")]),
    ]
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in needed:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name in needed for name in names):
                selected.append(node)

    test_module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {}
    exec(compile(test_module, "<extract_python_code>", "exec"), namespace)
    return namespace["extract_python_code"]


extract_python_code = _load_extract_python_code()
code_extraction_max_retries = extract_python_code.__globals__["code_extraction_max_retries"]
code_retry_user_prompt = extract_python_code.__globals__["code_retry_user_prompt"]
anthropic_messages_enabled = extract_python_code.__globals__["anthropic_messages_enabled"]
llm_model_name = extract_python_code.__globals__["llm_model_name"]
anthropic_auth_header = extract_python_code.__globals__["anthropic_auth_header"]
anthropic_custom_headers = extract_python_code.__globals__["anthropic_custom_headers"]


class ExtractPythonCodeTests(unittest.TestCase):
    def test_strict_python_fence(self):
        text = "```python\nprint(1)\n```"
        self.assertEqual(extract_python_code(text), "print(1)")

    def test_case_and_space_variant(self):
        text = "Here:\n``` Python\nprint(2)\n```"
        self.assertEqual(extract_python_code(text), "print(2)")

    def test_py_alias_and_crlf(self):
        text = "```py\r\nprint(3)\r\n```"
        self.assertEqual(extract_python_code(text), "print(3)")

    def test_plain_fence(self):
        text = "```\nimport sympy as sp\nprint(sp.Integer(4))\n```"
        self.assertEqual(extract_python_code(text), "import sympy as sp\nprint(sp.Integer(4))")

    def test_unclosed_fence(self):
        text = "```python\nprint(5)"
        self.assertEqual(extract_python_code(text), "print(5)")

    def test_tilde_fence(self):
        text = "~~~python\nprint(6)\n~~~"
        self.assertEqual(extract_python_code(text), "print(6)")

    def test_multiple_fences_prefers_python(self):
        text = "```text\nnot executable prose\n```\n```python\nprint(7)\n```"
        self.assertEqual(extract_python_code(text), "print(7)")

    def test_bare_code_block_in_prose(self):
        text = "I will compute it directly.\nimport math\nx = math.sqrt(9)\nprint(x)\nThat is enough."
        self.assertEqual(extract_python_code(text), "import math\nx = math.sqrt(9)\nprint(x)")

    def test_thinking_text_is_not_code(self):
        text = (
            "Looking at this problem: we need the largest prime $p$ such that "
            "$J_0(p)$ is isogenous over $\\mathbb{Q}$ to a product of elliptic curves."
        )
        self.assertEqual(extract_python_code(text), "")

    def test_empty_fence_is_not_code(self):
        text = "```python\n\n```"
        self.assertEqual(extract_python_code(text), "")

    def test_fence_with_leading_thought_salvages_code(self):
        text = (
            "```python\n"
            "Looking at this problem, I need to evaluate the integral first.\n"
            "expr = 'Integrate[x^2,{x,0,1}]'\n"
            "print(wolfram_eval(expr))\n"
            "```"
        )
        self.assertEqual(
            extract_python_code(text),
            "expr = 'Integrate[x^2,{x,0,1}]'\nprint(wolfram_eval(expr))",
        )

    def test_bare_thought_before_code(self):
        text = (
            "Looking at this problem, I need to evaluate an integral.\n\n"
            "expr = 'Integrate[x^2,{x,0,1}]'\n"
            "print(wolfram_eval(expr))"
        )
        self.assertEqual(
            extract_python_code(text),
            "expr = 'Integrate[x^2,{x,0,1}]'\nprint(wolfram_eval(expr))",
        )

    def test_code_extraction_retry_config_defaults_to_one(self):
        old = os.environ.pop("CODE_EXTRACTION_MAX_RETRIES", None)
        try:
            self.assertEqual(code_extraction_max_retries(), 1)
        finally:
            if old is not None:
                os.environ["CODE_EXTRACTION_MAX_RETRIES"] = old

    def test_code_extraction_retry_config_is_bounded(self):
        old = os.environ.get("CODE_EXTRACTION_MAX_RETRIES")
        try:
            os.environ["CODE_EXTRACTION_MAX_RETRIES"] = "99"
            self.assertEqual(code_extraction_max_retries(), 3)
            os.environ["CODE_EXTRACTION_MAX_RETRIES"] = "-2"
            self.assertEqual(code_extraction_max_retries(), 0)
            os.environ["CODE_EXTRACTION_MAX_RETRIES"] = "not-an-int"
            self.assertEqual(code_extraction_max_retries(), 1)
        finally:
            if old is None:
                os.environ.pop("CODE_EXTRACTION_MAX_RETRIES", None)
            else:
                os.environ["CODE_EXTRACTION_MAX_RETRIES"] = old

    def test_code_retry_prompt_demands_python_only(self):
        prompt = code_retry_user_prompt("What is 1+1?", "Looking at this problem...")
        self.assertIn("executable Python code", prompt)
        self.assertIn("[TRAP_DETECTED]", prompt)
        self.assertIn("```python", prompt)
        self.assertIn("Looking at this problem", prompt)

    def test_anthropic_mode_helpers(self):
        old_values = {
            key: os.environ.get(key)
            for key in (
                "LLM_API_PROTOCOL",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_USER_AGENT",
            )
        }
        try:
            os.environ["LLM_API_PROTOCOL"] = "anthropic_messages"
            os.environ["OPENAI_API_KEY"] = "sk-test"
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ["ANTHROPIC_USER_AGENT"] = "claude-cli/2.0.76 (external, cli)"

            self.assertTrue(anthropic_messages_enabled())
            self.assertEqual(llm_model_name("PRO", "claude-opus-4-7"), "claude-opus-4-7")
            self.assertEqual(anthropic_auth_header(), "Bearer sk-test")
            self.assertEqual(
                anthropic_custom_headers(),
                {
                    "Authorization": "Bearer sk-test",
                    "User-Agent": "claude-cli/2.0.76 (external, cli)",
                },
            )
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
