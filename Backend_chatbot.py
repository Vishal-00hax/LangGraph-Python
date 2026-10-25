import os
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup() 

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    streaming=True,
)

SYSTEM_PROMPT = SystemMessage(content="You are a helpful assistant.")


class State(TypedDict):
    message: Annotated[list, add_messages]


def chat_node(state: State):
    response = llm.invoke([SYSTEM_PROMPT] + state["message"])
    return {"message": [response]}


graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    rows = conn.execute(
        "SELECT thread_id FROM checkpoints "
        "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
    ).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            break

        response = workflow.invoke(
            {"message": [HumanMessage(content=user_input)]}, config=config
        )
        print("AI:", response["message"][-1].content)