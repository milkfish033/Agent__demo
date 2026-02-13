import streamlit as st 
import time 
from rag import RagService
import config_data as config 

#title 
st.title("智能客服")
st.divider() 


if "msg" not in st.session_state:
    st.session_state["msg"] = [{"role": "assistant", "content": "你好,有什么可以帮你"}]


if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for msg in st.session_state["msg"]:
    st.chat_message(msg["role"]).write(msg["content"])


#在页面提供用户输入
prompt = st.chat_input()

if prompt:

    st.chat_message("user").write(prompt)
    st.session_state["msg"].append({"role": "user", "content": prompt})

    with st.spinner("助手思考中"):
        time.sleep(1)

    res_stream = st.session_state["rag"].chain.stream({"question": prompt}, config.session_config)
    res = st.chat_message("assistant").write_stream(res_stream)
    st.session_state["msg"].append({"role": "assistant", "content": res})

