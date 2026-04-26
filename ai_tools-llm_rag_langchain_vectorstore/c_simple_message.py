""" """

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from a_set_env_key import set_env


set_env()

# Ücretsiz açık kaynak model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
)

chat = ChatHuggingFace(llm=llm)

# Mesaj yapısı
messages = [
    SystemMessage(content="Sen uzman bir tarihçisin ve Türkçe yanıt veriyorsun."),
    HumanMessage(
        content="İstanbul'un fethinin dünya tarihindeki en büyük etkisi ne oldu?"
    ),
]

response = chat.invoke(messages)

print(response.content)
