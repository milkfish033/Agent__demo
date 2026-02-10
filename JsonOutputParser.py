from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate


js = JsonOutputParser()
s = StrOutputParser () 

model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我邻居姓{last_name}，刚生了个{gender}，帮我起名字？按照json格式返回，其中key是name， value是名字。"
    )

second_prompt = PromptTemplate.from_template(
    "姓名是{name}，请分析一下这个名字的含义？"
)

chain = first_prompt | model | js | second_prompt | model | s

res = chain.stream({"last_name": "李", "gender": "男孩"})

for chunk in res:
    print(chunk, end="", flush=True)