from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool 
from dotenv import load_dotenv 

load_dotenv()


@tool(description= "获取体重，返回值是整数，单位是千克")
def get_weight(name: str) -> str:
    return 90 


@tool(description= "获取身高，返回值是整数，单位是厘米")
def get_height(name: str) -> str:
    return 172


agent = create_agent (
    model = ChatTongyi(model = "qwen3-max"),
    tools = [get_height, get_weight],
    system_prompt= 
    """
    你是严格遵循ReAct框架的智能体，必须按照思考，行动，观察，再思考的逻辑解决问题。
    每轮最多调用一个工具，禁止单次调用多个工具。
    并且告诉我你的思考过程， 工具的调用原因。按思考，行动，观察三个方面告诉我。

    """
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "我的BMI是多少"}]},
    stream_mode = "values"
):
    lastest_msg = chunk["messages"][-1]
    if lastest_msg.content:
        print(type(lastest_msg).__name__, lastest_msg.content)
    try:
        if lastest_msg.tool_calls:
            print(f"工具调用 : {[tc ['name'] for tc in lastest_msg.tool_calls]}")
    except AttributeError as e:
        pass