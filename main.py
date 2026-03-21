from model import agent

result = agent.invoke(
    {"messages": [{"role": "user", "content": "中国最近发生了什么大事"}]}
)
print(result["messages"][-1].content)