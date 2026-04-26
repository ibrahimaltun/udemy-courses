import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from A_IntroTopics.a_set_env_key import set_env

set_env()

# langchain tools
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Model Initialize
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
)

llm_model = ChatHuggingFace(llm=endpoint)

# 2. Model Storage
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 3. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 4. Chain
chain = prompt | llm_model

config = {"configurable": {"session_id": "abcde123"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("[You]: ")
        except KeyboardInterrupt:
            print("\nThe system is closing...")
            break

        response = with_message_history.invoke(
            [HumanMessage(content=user_input)], config=config
        )
        print("[AI]: ", response.content)
