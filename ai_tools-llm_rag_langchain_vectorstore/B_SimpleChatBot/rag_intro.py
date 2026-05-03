import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from A_IntroTopics.a_set_env_key import set_env

set_env()

import bs4
from langchain import hub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# Model
end_point = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=end_point)

# loader
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

# split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# vector store
embedding = embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)


retriver = vectorstore.as_retriever()


# rag prompt
prompt = hub.pull("rlm/rag-prompt")
# OR
# prompt = """
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:
# """


def format_docs(doc):
    return "\n\n".join(d.page_content for d in doc)


# CHAIN
rag_chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("\n")
    # for chunk in rag_chain.stream("what is maximum inner product searc?"):
    for chunk in rag_chain.stream("what is the test the decomposition?"):
        print(chunk, end="", flush=True)
