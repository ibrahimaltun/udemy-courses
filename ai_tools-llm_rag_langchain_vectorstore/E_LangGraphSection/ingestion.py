import sys
import os

# ---------------------  SET ENV   ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from A_IntroTopics.a_set_env_key import set_env

set_env()

# ---------------------  Imports ---------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)

# -----------------  HuggingFace MODEL ---------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,  # Set to False for more stable tool calling
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)

# -----------------  Document Operations ---------------------

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

splits = text_splitter.split_documents(docs_list)

# -----------------  HuggingFace Embedding ---------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": "gpu"}
)

vectorstore = Chroma.from_documents(
    documents=docs_list,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma", embedding=embeddings, persist_directory="./.chroma"
).as_retriever()
