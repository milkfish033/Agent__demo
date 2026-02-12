from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

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

input_query = "何勇的职业是什么？"
reference = "["
res = vector_store.similarity_search(input_query, k=3)
for doc in res:
    reference += doc.page_content + "\n"

reference += "]"

def print_prompt(prompt):
    print("Prompt:", prompt.to_string())
    return prompt

chain = prompt | print_prompt | model | str_parser

response = chain.stream({"context": reference, "query": input_query})
for r in response:
    print(r, end="", flush=True)