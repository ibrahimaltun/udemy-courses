import sys
import os

# ---------------------  SET ENV   ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from A_IntroTopics.a_set_env_key import set_env

set_env()

# ---------------------  Imports ---------------------
# from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

# from langgraph.prebuilt import create_react_agent # that is deprecated
from langchain.agents import create_agent

# -----------------  HuggingFace MODEL ---------------------
end_point = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,  # Set to False for more stable tool calling
)

llm_model = ChatHuggingFace(llm=end_point)

# -----------------  Tavily Search ---------------------
# search = TavilySearchResults(max_results=2)
search = TavilySearch(max_results=2)

tools = [search]

# model_with_tools = llm_model.bind_tools(tools)

# ----------------- AGENT EXECUTOR ---------------------
# aget_executor = create_react_agent()
agent_executor = create_agent(llm_model, tools)

# ----------------- MAIN EXECUTION ---------------------
if __name__ == "__main__":
    # search_results = search.invoke("what is the weather in istanbul today?")
    # print(search_results)

    # To see tool calling in action, ask a question that requires a search
    query = "What is the weather in Istanbul now?"
    # response = model_with_tools.invoke([HumanMessage(content=query)])

    # print("Content:", response.content)
    # print("Tool Calls:", response.tool_calls)

    # WITH AGENT
    response = agent_executor.invoke({"messages": HumanMessage(content=query)})

    # print(response)
    for r in response["messages"]:
        print(r.content)
