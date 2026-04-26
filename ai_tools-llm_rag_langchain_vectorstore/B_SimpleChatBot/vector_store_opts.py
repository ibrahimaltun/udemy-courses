import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from A_IntroTopics.a_set_env_key import set_env

set_env()

# langchain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Vector Database
from langchain_chroma import Chroma

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


docs = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Vector store
vec_store = Chroma.from_documents(documents=docs, embedding=HuggingFaceEmbeddings())

# Retriever
retriever = RunnableLambda(vec_store.similarity_search()).bind(k=1)

# Model
end_point = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=end_point)

# Custom Message
messages = """
Answer this question using the provided context only.
{question}
Context:
{context}
"""

# Prompt
prompt = ChatPromptTemplate.from_messages([("human", messages)])

# Chain
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

if __name__ == "__main__":
    response = chain.invoke("tell me about cats")
    print(response.content)
