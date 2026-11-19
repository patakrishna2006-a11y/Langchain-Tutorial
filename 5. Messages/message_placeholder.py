from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='Chat_history'),
    ('human', '{query}')
])

chat_history = []
with open('chat_history.txt') as f: 
    chat_history.extend(f.readlines())

Prompt = chat_prompt.invoke({'Chat_history': chat_history, 'query': 'What is my refund?'})
result = llm.invoke(Prompt)
print(result.text)