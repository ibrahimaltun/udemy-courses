from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace

from d_simp_msg_wparser import llm_endpoint
from a_set_env_key import set_env

set_env()

llm_model = ChatHuggingFace(llm=llm_endpoint)

# messages = [
#     SystemMessage(content="Translate the following from English to Spanish."),
#     HumanMessage(content="Hi"),
# ]

system_prompt = "Translate the following into {language}"
prompt_templates = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{text}")]
)

parser = StrOutputParser()

# chain logic
chain = prompt_templates | llm_model | parser

if __name__ == "__main__":
    print(chain.invoke({"language": "Turkish", "text": "Hello world"}))
