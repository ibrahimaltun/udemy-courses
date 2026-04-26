"""
This code block set the environmental variable called HUGGINGFACEHUB_API_TOKEN
to use Hugging Face model which is free. This step gets the key from .env file.
"""

import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()
HGG_F_KEY = dotenv_values(".env").get("HUGGINGFACE_API_KEY")
# print(HGG_F_KEY)


def set_env():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HGG_F_KEY
