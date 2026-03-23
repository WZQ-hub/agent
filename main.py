from model import agent



full_message = None
chunks = []
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "世界最近发生了什么大事"}]},
    stream_mode="messages",
    version="v2",
):
    # 不需要判断 ["type"]，因为流里全是 messages
    print(f"metadata: {metadata}")
    print(f"token: {token}")