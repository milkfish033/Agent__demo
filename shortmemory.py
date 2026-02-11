import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template("你需要根据会话历史回应用户的问题。会话历史如下：\n{history}\n用户的问题是：{question}\n请根据会话历史回答用户的问题。")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回应用户的问题。"),
        MessagesPlaceholder("history"),
        ("human", "用户的问题是：{question}"),
    ]
)


str_parser = StrOutputParser()

def print_prompt(input: dict):
    print("="*20, input, "="*20)
    return input


#creare a new chain with memory
base_chain = prompt | print_prompt | model | str_parser



store = {}
#get memory based on session id
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history, # get InmemoryChatMessageHistory class by session id 
    input_messages_key= "question", #用户输入在模版中的占位符
    history_messages_key= "history" #会话历史在模版中的占位符
)

if __name__ == "__main__":
    #固定格式，创建session id
    session_config = {
        "configurable" : {
            "session_id": "user_12345"
        }
    }
    res = conversation_chain.invoke({"question": "小明有两个猫"}, session_config)
    print("first response: ", res)

    res = conversation_chain.invoke({"question": "小王有四个老虎"}, session_config)
    print("second response: ", res)

    res = conversation_chain.invoke({"question": "一共有几个宠物"}, session_config)
    print("third response: ", res)

