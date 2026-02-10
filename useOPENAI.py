import os 
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个python变成专家"},
        {"role": "assistant", "content": "我是编程专家，你要问什么"},
        {"role": "user", "content": "输出1-10使用代码"}
    ],
    stream = True
)

for chunk in response:
    print(chunk.choices[0].delta.content,
          end = "", #每一段之间以空格分局
          flush = True #立刻刷新缓冲区
    )
    
