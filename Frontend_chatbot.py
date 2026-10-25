import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from Backend_chatbot import workflow, checkpointer, get_all_threads

st.set_page_config(page_title="My Chatbot", page_icon="💬")
st.title("My Chatbot")


def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def switch_chat(thread_id):
    state = workflow.get_state({"configurable": {"thread_id": thread_id}})
    msgs = state.values.get("message", [])
    st.session_state.thread_id = thread_id
    st.session_state.messages = [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in msgs
    ]


def delete_chat(thread_id):
    checkpointer.delete_thread(thread_id)
    if thread_id == st.session_state.thread_id:
        remaining = get_all_threads()
        if remaining:
            switch_chat(remaining[0])
        else:
            new_chat()



if "thread_id" not in st.session_state:
    new_chat()

# Sidebar
with st.sidebar:
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)
    st.caption("Current thread ID")
    st.code(st.session_state.thread_id, language=None)

    st.divider()
    st.caption("Previous chats")
    for tid in get_all_threads():
        col_open, col_del = st.columns([4, 1])
        label = ("▶ " if tid == st.session_state.thread_id else "") + tid[:8]
        col_open.button(label, key=f"open_{tid}", on_click=switch_chat, args=(tid,), use_container_width=True)
        col_del.button("🗑", key=f"del_{tid}", on_click=delete_chat, args=(tid,))

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