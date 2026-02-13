from langchain_chroma import Chroma
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv 

load_dotenv()

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        embedding: 嵌入模型的传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name = config.collection_name, #数据库的表名
            embedding_function= self.embedding,
            persist_directory= config.persist_directory #向量数据库保存路径
        )

    def get_retriever(self):
        """
        获取向量数据库的检索器
        top_k: 检索时返回的最相似的文本数量
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": config.similarity_top_k})
        return retriever



if __name__ == "__main__":
    embedding = DashScopeEmbeddings(model = "text-embedding-v4")
    vector_store_service = VectorStoreService(embedding)
    retriever = vector_store_service.get_retriever()
    res = retriever.invoke("沈明宇的专业是什么？")
    print(res)