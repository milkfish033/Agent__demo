from langchain_community.llms.tongyi import Tongyi
import os 


model = Tongyi (model = "qwen-max", api_key = os.getenv("DASHSCOPE_API_KEY"))

res = model.stream(input = "请介绍一下你自己")

for chunk in res:
    print(chunk, end="", flush=True)