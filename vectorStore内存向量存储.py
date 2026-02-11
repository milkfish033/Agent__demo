from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv

load_dotenv()
vector_store = InMemoryVectorStore(
    embedding= DashScopeEmbeddings()

)

loader = CSVLoader(
    file_path="./data/qa.csv", 
    encoding="utf-8",
    source_column= "name"
    )

documents = loader.load()

#新增
vector_store.add_documents(
    documents = documents,
    ids = ["id" + str(i) for i in range(1, len(documents) + 1)] #给添加的文档提供id
    ) 

#删除
vector_store.delete(
    ids = ["id1", "id2"] #根据id删除文档
    )

#检索
res = vector_store.similarity_search(
    query = "score",
    k = 3 #返回最相似的3条文档
    )

print(res)