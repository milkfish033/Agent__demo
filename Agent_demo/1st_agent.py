from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv 
from langchain_core.tools import tool
load_dotenv()


@tool(description= "查询天气")
def get_weather() -> str:
    return "深圳的天气是晴天"



agent = create_agent(
    model = ChatTongyi(model = 'qwen3-max'),
    tools = [get_weather],
    system_prompt= "你是一个聊天助手，你的任务是回答问题"
)

res = agent.invoke(
    {
        "messages" :[
            {"role": "user", "content": "明天深圳的天气如何"}
        ]
    }
)

for msg in res["messages"]:
    print(type(msg).__name__, msg.content)