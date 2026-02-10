from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template("请用中文回答：{question}")

parser = StrOutputParser()

chain = prompt | model | parser | model 
res = chain.invoke({"question": "介绍一下LangChain"})
print(res.content)