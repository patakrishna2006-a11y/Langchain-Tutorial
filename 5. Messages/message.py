from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Langchain.")
]

result = llm.invoke(messages)

messages.append(AIMessage(content = result.text))

print(messages)