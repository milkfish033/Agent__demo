from langchain_community.chat_models.tongyi import ChatTongyi 

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os 

chat = ChatTongyi(model="qwen3-max", api_key=os.getenv("DASHSCOPE_API_KEY"))

# msg = [
#     SystemMessage(content="你是一个诗歌创作专家,你的风格是热烈奔放的"),
#     HumanMessage(content="写一首诗"),
#     AIMessage(content="好的，请稍等"),
#     HumanMessage(content="春天来了，花儿开了，鸟儿唱了，请继续创作一首诗"),
# ]

style = "热烈奔放的"
msg = [
    ("system", f"你是一个诗歌创作专家,你的风格是{style}"),
    ("human", "写一首诗"),
    ("ai", "好的，请稍等"),
    ("human", "春天来了，花儿开了，鸟儿唱了，请继续创作一首诗"),
]
# res = chat.stream(msg)  

# for chunk in res:
    # print(chunk.content, end="", flush=True)    


from langchain_community.embeddings import DashScopeEmbeddings

embed = DashScopeEmbeddings()

print(embed.embed_query("你好，百炼"))

# print("------------------")
# print(embed.embed_documents(["今天天气不错", "我喜欢吃苹果"]))