import sys
import os

# ---------------------  SET ENV   ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from A_IntroTopics.a_set_env_key import set_env

set_env()

# ---------------------  Imports ---------------------

from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain.agents import create_agent

from langgraph.checkpoint.sqlite import SqliteSaver

# -----------------  Tavily Search ---------------------
search_tool = TavilySearch(
    max_results=2,
)
tools = [search_tool]


# -----------------  HuggingFace MODEL ---------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,  # Set to False for more stable tool calling
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)

# ------------- AGENT with MEMORY
with SqliteSaver.from_conn_string("agent_history.sqlite") as memory:
    # ----------------- AGENT EXECUTOR ---------------------
    agent_executor = create_agent(chat_model, tools)

    # Configurations for memory usage with session/thread id
    config = {"configurable": {"thread_id": "user_session_1"}}

    print("--- AI Agent Started (Type 'exit' to stop) ---")

    while True:
        # Get real input from the terminal
        user_input = input("User: ")

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)

        # Streaming the response chunks
        for chunk in agent_executor.stream(
            {"messages": [("user", user_input)]}, config, stream_mode="messages"
        ):
            # The 'messages' stream mode yields (message, metadata) tuples
            message, metadata = chunk

            # Print content as it streams from the LLM
            if message.content:
                print(message.content, end="", flush=True)

        print("\n" + "-" * 20)
