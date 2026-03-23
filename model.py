import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_openai import ChatOpenAI
import dotenv
from tools import *
from langchain.messages import ToolMessage


dotenv.load_dotenv()
api_key = os.getenv("API_TOKEN")

tools = [get_weather, search_news]


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model base on conversation complexity"""
    message_count = len(request.state["messages"])



    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))


advanced_model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    base_url="https://llm.chutes.ai/v1",
    api_key=api_key,
    temperature=0.1,
)

basic_model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324-TEE",
    base_url="https://llm.chutes.ai/v1",
    api_key=api_key,
    temperature=0.1,
)
agent = create_agent(
    model = advanced_model,
    middleware=[dynamic_model_selection],
    tools=tools,
    system_prompt="你是一个新闻助手， 你需要仔细分析和验证新闻的准确率"
)
