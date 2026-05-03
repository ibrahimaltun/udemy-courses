"""
This code block set the environmental variable called HUGGINGFACEHUB_API_TOKEN
to use Hugging Face model which is free. This step gets the key from .env file.
"""

import os
from dotenv import dotenv_values

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

HGG_F_KEY = dotenv_values(dotenv_path=env_path).get("HUGGINGFACE_API_KEY")
# print(HGG_F_KEY)
TAVILY_KEY = dotenv_values(dotenv_path=env_path).get("TAVILY_API_KEY")


def set_env():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HGG_F_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_KEY
