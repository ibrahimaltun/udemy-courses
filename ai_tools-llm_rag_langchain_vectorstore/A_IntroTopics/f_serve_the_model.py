from a_set_env_key import set_env
from e_simp_msg_wtemplates import chain

# FAST API
from fastapi import FastAPI

from langserve import add_routes
# from pydantic import BaseModel, Field

set_env()


# class TranslationInput(BaseModel):
#     language: str = Field(description="Hedef dil (Örn: Spanish, Turkish)")
#     text: str = Field(description="Çevrilecek metin")


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Hugging Face ve LangServe API",
)

# 3. LangServe rotasını ekle
add_routes(
    app,
    chain, #.with_types(input_type=TranslationInput),
    path="/translate",
)

if __name__ == "__main__":
    print("a -> llm model::: ", chain)
    print("*" * 50)
    print(app)
    import uvicorn

    uvicorn.run(app, host="localhost", port=8009)
