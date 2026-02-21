"""
ReWOO — Reasoning Without Observation
======================================
架构：Planner → Worker → Solver（线性三段式）

LLM 显式调用次数：2（Planner 1次 + Solver 1次）
工具调用次数：O(k)，k 为规划步骤数
"""

import re
import ast
import math
import operator as op
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, START, END

from smolagents import WebSearchTool

from utils.prompt_loader import planner_prompt, solver_prompt


# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatTongyi(model="qwen3-max")


# ── Tools ─────────────────────────────────────────────────────────────────────

_web_search_tool = WebSearchTool()

_wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
)

# 安全计算器允许的运算符
_SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub,
    ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow,  ast.USub: op.neg,
}
_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def _web_search(query: str) -> str:
    return str(_web_search_tool(query))


def _wikipedia(query: str) -> str:
    return _wiki.run(query)


def _llm_reason(prompt: str) -> str:
    return llm.invoke([HumanMessage(content=prompt)]).content


def _calculator(expression: str) -> str:
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _SAFE_NAMES:
                return _SAFE_NAMES[node.id]
            raise ValueError(f"不允许的名称: {node.id}")
        if isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式类型: {ast.dump(node)}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"计算错误: {e}"


TOOL_MAP = {
    "web_search": _web_search,
    "Wikipedia":  _wikipedia,
    "LLM":        _llm_reason,
    "Calculator": _calculator,
}


# ── State ─────────────────────────────────────────────────────────────────────

class ReWOOState(TypedDict):
    task:        str   # 原始用户任务
    plan_string: str   # Planner 原始输出文本
    steps:       list  # [(plan_desc, "#E1", "tool_name", "tool_input"), ...]
    results:     dict  # {"#E1": "actual result", ...}
    result:      str   # Solver 最终答案


# ── Helpers ───────────────────────────────────────────────────────────────────

_PLAN_RE = re.compile(
    r"Plan:\s*(.+?)\n#(E\d+)\s*=\s*(\w+)\[(.+?)\]",
    re.DOTALL,
)


def _clean_plan(text: str) -> str:
    """去掉 qwen3 的 <think>...</think> 块和 markdown 代码围栏。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[^\n]*\n?", "", text)
    return text.strip()


def _parse_plan(plan_string: str) -> list:
    """将 Planner 输出解析为步骤列表。

    Returns:
        list of (plan_desc, "#E1", "tool_name", "tool_input")
    """
    return [
        (plan.strip(), f"#{var}", tool.strip(), inp.strip())
        for plan, var, tool, inp in _PLAN_RE.findall(plan_string)
    ]


def _substitute(text: str, results: dict) -> str:
    """将 text 中的 #Ei 引用替换为 results 中的实际值。"""
    for var, val in results.items():
        text = text.replace(var, val)
    return text


# ── Nodes ─────────────────────────────────────────────────────────────────────

def planner_node(state: ReWOOState) -> dict:
    """一次性生成包含所有步骤的完整执行蓝图（DAG）。"""
    response = llm.invoke([
        SystemMessage(content=planner_prompt),
        HumanMessage(content=state["task"]),
    ])
    plan_string = _clean_plan(response.content)
    steps = _parse_plan(plan_string)
    return {
        "plan_string": plan_string,
        "steps": steps,
        "results": {},
    }


def _get_deps(tool_input: str) -> set:
    """提取 tool_input 中引用的所有 #Ei 变量名。"""
    return set(re.findall(r"#E\d+", tool_input))


def worker_node(state: ReWOOState) -> dict:
    """按 DAG 依赖关系执行工具，同一批次内并行，跨批次顺序。

    算法：
      1. 找出当前所有依赖已满足的步骤（ready wave）
      2. 用 ThreadPoolExecutor 并行执行这一批
      3. 将结果写入 results，重复直到所有步骤完成
    """
    # 以 variable 为 key 建立索引
    step_map = {var: (tool, inp) for _, var, tool, inp in state["steps"]}
    # 保持 Planner 原始顺序（用于 Solver 整合时按序展示）
    order = [var for _, var, _, _ in state["steps"]]

    results: dict = {}
    pending = list(order)

    while pending:
        # 当前轮次中依赖已全部就绪的步骤
        ready = [
            var for var in pending
            if _get_deps(step_map[var][1]).issubset(results.keys())
        ]

        def _run(var: str) -> tuple[str, str]:
            tool_name, tool_input = step_map[var]
            resolved = _substitute(tool_input, results)
            fn = TOOL_MAP.get(tool_name, lambda q: f"[未知工具: {tool_name}，输入: {q}]")
            return var, str(fn(resolved))

        with ThreadPoolExecutor(max_workers=len(ready)) as executor:
            futures = {executor.submit(_run, var): var for var in ready}
            for future in as_completed(futures):
                var, output = future.result()
                results[var] = output
                print(f"  {var} [{step_map[var][0]}] → {output[:150]}")

        for var in ready:
            pending.remove(var)

    return {"results": results}


def solver_node(state: ReWOOState) -> dict:
    """整合所有 (Plan_i, Evidence_i) 对，输出最终答案。"""
    evidence_parts = []
    for plan_desc, variable, _, _ in state["steps"]:
        evidence = state["results"].get(variable, "[无结果]")
        evidence_parts.append(f"计划: {plan_desc}\n{variable}: {evidence}")

    evidence_text = "\n\n".join(evidence_parts)
    user_msg = (
        f"任务: {state['task']}\n\n"
        f"执行结果:\n{evidence_text}\n\n"
        f"请给出最终答案:"
    )

    response = llm.invoke([
        SystemMessage(content=solver_prompt),
        HumanMessage(content=user_msg),
    ])
    return {"result": response.content}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_rewoo():
    """构建 ReWOO 线性图：START → planner → worker → solver → END"""
    builder = StateGraph(ReWOOState)
    builder.add_node("planner", planner_node)
    builder.add_node("worker",  worker_node)
    builder.add_node("solver",  solver_node)

    builder.add_edge(START,     "planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker",  "solver")
    builder.add_edge("solver",  END)

    return builder.compile()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_rewoo()

    task = "分析2026年美股市场走向"

    print("=" * 60)
    print(f"任务: {task}")
    print("=" * 60)

    result = graph.invoke({"task": task})

    print("\n── Planner 蓝图 ──")
    print(result["plan_string"])

    print("\n── Worker 执行结果 ──")
    for var, val in result["results"].items():
        print(f"  {var}: {val[:300]}")

    print("\n── Solver 最终答案 ──")
    print(result["result"])
    print("=" * 60)
