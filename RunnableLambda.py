from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi  
from langchain_core.runnables import RunnableLambda

model = ChatTongyi(model="qwen3-max")
str_parser = StrOutputParser() 

first_prompt = PromptTemplate.from_template(
    "我邻居姓{last_name}，刚生了个{gender}，帮我起名字？只返回名字。"
)

second_prompt = PromptTemplate.from_template(
    "姓名是{name}，请分析一下这个名字的含义？"
)

#ai msg -> dict ({"name": ai msg})
# my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})


chain = first_prompt | model | (lambda x: {"name": x.content}) | second_prompt | model | str_parser

for chunk in chain.stream({"last_name": "董", "gender": "男孩"}):
    print(chunk, end="", flush=True)