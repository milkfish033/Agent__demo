from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

str_parser = StrOutputParser()
model = ChatTongyi(model = 'qwen3-max')

prompt = ChatPromptTemplate.from_messages([
    ("system", "按照我给你的资料，回答用户问题。参考资料：{context}"),
    ("human", "用户提问：{query}")
])

vector_store = Chroma(
    collection_name = "my_collection",
    embedding_function= DashScopeEmbeddings(),
    persist_directory= "./chroma_db"

)

#完成检索入链
#在chain中每个对象都必须是runnable
#langchain中向量储存储存对象 有一个方法 as_runnable() 可以将其转换为runnable对象

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

input_query = "何勇的职业是什么？"


def print_prompt(prompt):
    print("Prompt:", prompt.to_string())
    return prompt


def format_func(docs):
    if not docs:
        return "没有找到相关资料。"
    formatted = "以下是相关资料：\n"
    for i, doc in enumerate(docs, 1):
        formatted += f"{i}. {doc.page_content}\n"
    return formatted

chain =  (
    {"query": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | str_parser
)

#这里直接传入text，因为retriever需要对text对embedding
for chunk in chain.stream(input_query):
    print(chunk, end="", flush=True)