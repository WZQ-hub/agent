from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

SYSTEM_PROMPT = """你是一位专业的天气预报专家，说话时喜欢使用双关语或有趣的天气相关玩笑。

你可以使用两个工具：

- get_weather_for_location：用于获取某个具体地点的天气
- get_user_location：用于获取用户当前所在的位置

如果用户询问天气，你必须先确定地点。
如果从问题中可以判断用户指的是“他们当前所在的位置”，请使用 get_user_location 工具获取用户的位置。"""

@tool()
def get_weather(city: str) -> str:
    '''可以根据城市名称获取天气信息'''
    return f"It's always sunny in {city}"

@dataclass()
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool()
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    '''根据用户 ID 获取用户所在的位置'''
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "上海"

model = init_chat_model(
    "deepseek-chat",
    temperature=0,
)

#define response format
@dataclass()
class ResponseFormat:
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

checkpointer = InMemorySaver()
# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
