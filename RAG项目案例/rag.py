from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings 
import config_data as config 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_history_store import FileChatMessageHistory, get_history
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
load_dotenv()   


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(embedding = DashScopeEmbeddings(model = config.embedding_name)) #向量数据库服务实例对象

        self.chat_model = ChatTongyi(model = config.chat_model_name) #聊天模型实例对象

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是我的人工智能助手，根据我提供的信息回答相应问题。参考资料如下：{context}"),
                ("system", "并且我提供用户的会话历史记录， 如下："),
                MessagesPlaceholder(variable_name="history"),
                ("user", "请回答用户问题：{question}"),
            ]
        )   

        self.chain = self._get_chain()

    
    def _get_chain(self):
        """
        获取最终的执行链
        """
        retriever = self.vector_service.get_retriever() #获取向量数据库的检索器

        def format_document(docs: list[Document]) -> list:
            if not docs:
                return "没有相关资料。"
            
            formatted_docs = ""
            for doc in docs:
                formatted_docs += f"文档片段: {doc.page_content}\n文档元数据;{doc.metadata}\n\n"

            return formatted_docs
        
        def print_prompt(prompt):
            print("="*30)
            print(prompt.to_string())
            print("="*30)
            return prompt
            

        def tmp(value):
            return value["question"]
            
        def tmp2(value):
            new_dict = {}
            new_dict["question"] = value["question"]["question"]
            new_dict["context"] = value["context"]
            new_dict["history"] = value["question"]["history"]
            return new_dict

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(tmp) | retriever | format_document
            } | RunnableLambda(tmp2) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

    
    
        #chain with history 
        conversation_chain = RunnableWithMessageHistory(
            chain, 
            get_history,
            input_messages_key="question", 
            history_messages_key= "history"
            )
        

        return conversation_chain


if __name__ == "__main__":
    # res = RagService().chain.invoke("沈明宇的做过哪些项目")
    # print(res)

    #session_id 
    session_config = {
        "configurable": {
            "session_id": "admin"
        }
    }

    res = RagService()._get_chain().invoke(
        {"question": "沈明宇的专业是什么？"},
        session_config
    )
    print(res)