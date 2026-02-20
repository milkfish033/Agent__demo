import operator
from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_community.chat_models.tongyi import ChatTongyi

# 1) Shared state across parent + subgraphs
class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    tasks: Annotated[list[str], operator.add]
    artifacts: Annotated[list[str], operator.add]
    current_task: NotRequired[str]
    next_node: NotRequired[Literal["research", "write", "finalize"]]


# 2) Sub-agent shared tools
@tool
def add_task(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """将任务追加到任务列表中。"""
    return Command(update={
        "tasks": [task],
        "messages": [ToolMessage(f"任务已添加：{task}", tool_call_id=tool_call_id)],
    })


@tool
def save_artifact(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """将内容（笔记、片段、参考资料）保存到短期记忆中。"""
    return Command(update={
        "artifacts": [content],
        "messages": [ToolMessage("成果已保存。", tool_call_id=tool_call_id)],
    })


# 3) Planner — custom node with structured output (one decision per turn)
class PlannerDecision(BaseModel):
    reasoning: str = Field(description="用中文说明当前进度和决策依据")
    action: Literal["research", "write", "finalize"] = Field(description="下一步行动")
    new_tasks: list[str] = Field(default=[], description="新增任务（仅 action=research 时填写）")


def build_planner():
    model = ChatTongyi(model="qwen3-max")
    decision_model = model.with_structured_output(PlannerDecision)

    def planner_node(state: OrchestratorState):
        artifacts = state.get("artifacts", [])
        tasks = state.get("tasks", [])
        current_stage = state.get("next_node")  # 记录上一轮路由到哪里
        messages = list(state["messages"])

        system = SystemMessage(content=(
            "你是一个规划者，负责制定执行计划并根据子智能体的结果动态调整。\n"
            f"当前执行阶段：{current_stage or '初始（尚未开始）'}\n"
            f"当前任务列表：{tasks}\n"
            f"已保存成果数：{len(artifacts)}\n"
            "决策规则（严格按阶段判断，不要分析消息内容）：\n"
            "1. 若阶段为 None/初始 → 选择 'research'，在 new_tasks 中制定2-3个调研子任务\n"
            "2. 若阶段为 'research'（调研刚完成）→ 评估调研结果，若符合预期则选择 'write'，\n"
            "   若不符合可更新 new_tasks 并再次选择 'research'\n"
            "3. 若阶段为 'write'（写作刚完成）→ 评估写作结果，符合预期则选择 'finalize'，\n"
            "   否则可再次选择 'write'\n"
            "请用中文填写 reasoning，说明对上一步结果的评估和下一步决策。"
        ))

        decision = decision_model.invoke([system] + messages)
        update = {
            "next_node": decision.action,
            "messages": [AIMessage(content=f"[规划] {decision.reasoning}")],
        }
        if decision.new_tasks:
            update["tasks"] = decision.new_tasks
        return update

    return planner_node


# 4) Research agent (simple stub tools)
@tool
def web_search(query: str) -> str:
    """网络搜索（存根）；如需真实集成请替换实现。"""
    return f"[搜索结果] 关于'{query}'的摘要内容。"


@dynamic_prompt
def research_prompt(request):
    return (
        "你是一个研究助手，负责使用可用工具进行调研并生成简洁笔记。\n"
        "完成后，用中文向规划者总结调研结果。"
    )


def build_research_agent():
    model  = ChatTongyi(model="qwen3-max")
    return create_agent(
        model=model,
        tools=[web_search, save_artifact],  # can save notes back into state
        middleware=[research_prompt],
        state_schema=OrchestratorState,
        name="research_agent",
    )


# 5) Writer agent (simple stub tools)
@tool
def draft_section(topic: str) -> str:
    """为最终输出起草一个简短章节。"""
    return f"[草稿] 关于'{topic}'的简洁章节内容。"


@dynamic_prompt
def writer_prompt(request):
    return (
        "你是一个写作助手，负责根据任务、成果和消息记录起草简洁内容。\n"
        "请用中文撰写简短章节，避免冗余。"
    )


def build_writer_agent():
    model  = ChatTongyi(model="qwen3-max")
    return create_agent(
        model=model,
        tools=[draft_section, save_artifact],
        middleware=[writer_prompt],
        state_schema=OrchestratorState,
        name="writer_agent",
    )


# 6) Parent orchestrator graph wiring
def build_plan_tasks_and_execute():
    planner = build_planner()
    researcher = build_research_agent()
    writer = build_writer_agent()

    builder = StateGraph(OrchestratorState)
    builder.add_node("planner", planner)
    builder.add_node("research_agent", researcher)
    builder.add_node("writer_agent", writer)

    def route_from_planner(state: OrchestratorState):
        action = state.get("next_node")
        if action == "research":
            return "research_agent"
        if action == "write":
            return "writer_agent"
        return END

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_from_planner, ["research_agent", "writer_agent", END])
    builder.add_edge("research_agent", "planner")
    builder.add_edge("writer_agent", "planner")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_plan_tasks_and_execute()
    config = {"configurable": {"thread_id": "demo-1"}}

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": "目标：针对'美股市场2026行情'生成一份包含3个要点的简短总结。先调研，再撰写。",
            }
        ],
        "tasks": [],
        "artifacts": [],
    }

    seen_ids: set = set()

    def print_new_messages(msgs):
        for msg in msgs:
            msg_id = getattr(msg, "id", None) or id(msg)
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)
            role = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", "")
            if content:
                print(f"  [{role}] {content}")

    print("=" * 50)
    for step in graph.stream(inputs, config, stream_mode="updates"):
        for node_name, update in step.items():
            msgs = update.get("messages", [])
            new_msgs = [m for m in msgs if (getattr(m, "id", None) or id(m)) not in seen_ids]
            has_extra = any(k in update for k in ("tasks", "artifacts", "next_node"))
            if not new_msgs and not has_extra:
                continue
            print(f"\n>>> 节点：{node_name}")
            print_new_messages(msgs)
            if "tasks" in update and update["tasks"]:
                print(f"  [任务列表] {update['tasks']}")
            if "artifacts" in update and update["artifacts"]:
                print(f"  [成果] {update['artifacts']}")
            if "next_node" in update:
                print(f"  [下一步] → {update['next_node']}")
    print("\n" + "=" * 50)
    print("流程结束")