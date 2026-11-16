# Google Gemini LLM 
from langchain_google_genai import GoogleGenerativeAI
import dotenv

dotenv.load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

result = llm.invoke("What is the capital of India?")

print(result)