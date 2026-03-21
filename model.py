import os

from langchain.agents import create_agent
from langchain_openai import OpenAI
import dotenv

dotenv.load_dotenv()
api_key = os.getenv("API_TOKEN")


model = OpenAI(
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    base_url="https://llm.chutes.ai/v1",
    api_key=api_key,
    temperature=0.1,
)

agent = create_agent(model)