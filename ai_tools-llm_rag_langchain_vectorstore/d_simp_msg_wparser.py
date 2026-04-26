from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from a_set_env_key import set_env

set_env()

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
)

llm_model = ChatHuggingFace(llm=llm_endpoint)

messages = [
    SystemMessage(content="Translate the following from English to Arabic."),
    HumanMessage(content="Hi my name is ibrahim, how are you today?"),
]

parser = StrOutputParser()

# chain logic
chain = llm_model | parser

if __name__ == "__main__":
    print(chain.invoke(messages))
