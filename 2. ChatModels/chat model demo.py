# Nvidia ChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

result = llm.invoke("write a 5 line poem about the love of indian teenage in a simple english.")
print(result.text)