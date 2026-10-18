import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from Backend_chatbot import workflow

st.set_page_config(page_title="My Chatbot", page_icon="💬")
st.title("My Chatbot")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    def token_stream():
        for chunk, meta in workflow.stream(
            {"message": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="messages",
        ):
            if meta.get("langgraph_node") == "chat_node" and isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(token_stream())
        except Exception as e:
            reply = ""
            st.error(f"Error: {e}")

        if not reply:
            st.warning("The model returned an empty response.")

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})