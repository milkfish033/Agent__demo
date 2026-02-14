from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool 
from dotenv import load_dotenv 

load_dotenv()


@tool(description= "传入股票名称，按照字符串的形式返回价格")
def get_price(name: str) -> str:
    return f"股票{name}的价格是20元"


@tool(description= "传入股票名称，获取该股票的上市公司的信息")
def get_info(name: str) -> str:
    return f"股票{name}是一家上市公司，专注计算机行业"


agent = create_agent (
    model = ChatTongyi(model = "qwen3-max"),
    tools = [get_price, get_info],
    system_prompt= "你是一名咨询助手，提供用户需要的信息，你可以使用提供的工具，但是需要告知我思考过程，解释你为什么选择某个工具"
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "小马发财是什么公司，股价是多少"}]},
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