from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
import os

chat_prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个诗歌创作专家,你的风格是热烈奔放的"),
        MessagesPlaceholder("history"),
        ("human", "写一首诗"),
    ]
)

history = [
    ("human", "春天来了，花儿开了，鸟儿唱了，请继续创作一首诗"),
    ("ai", "春眠不觉晓， 处处闻啼鸟。夜来风雨声，花落知多少。"),
    ("human", "好样的，再来一个"),
    ("ai", "春天的脚步近了，花儿开了，鸟儿唱了，春风吹绿了大地，万物复苏了，春天来了！"),
]


prompt_text = chat_prompt_template.invoke({"history": history}).to_string()
# print(prompt_text)

model = ChatTongyi(model="qwen3-max", api_key=os.getenv("DASHSCOPE_API_KEY"))
print(model.invoke(prompt_text).content)