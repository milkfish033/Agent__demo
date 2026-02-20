from langchain.agents import create_agent, AgentState
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool 
from dotenv import load_dotenv 
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call
from langgraph.runtime import Runtime
load_dotenv()


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[before agent], there are {len(state["messages"])}")


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after agent], there are {len(state["messages"])}")



@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[before model], there are {len(state["messages"])}")


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after model], there are {len(state["messages"])}")


@wrap_model_call
def model_call_hook(request, handler):
    print("model in use ")
    return handler(request)

@wrap_tool_call
def monitor_tool(request, handler):
    print(f"tool in use: {request.tool_call['name']}")
    print(f"tool in use: {request.tool_call['args']}")
    return handler(request)



@tool(description= "输入城市名称，查询天气")
def get_weather(city) -> str:
    return "f: {city}的天气是晴天"



agent = create_agent(
    model = ChatTongyi(model = 'qwen3-max'),
    tools = [get_weather],
    middleware= [log_after_agent, log_after_model, log_before_agent, log_before_model, model_call_hook, monitor_tool],
    system_prompt= "你是一个聊天助手，你的任务是回答问题"
)

res = agent.invoke(
    {
        "messages" :[
            {"role": "user", "content": "深圳的天气如何，怎么穿衣服"}
        ]
    }
)

for msg in res["messages"]:
    print(type(msg).__name__, msg.content)