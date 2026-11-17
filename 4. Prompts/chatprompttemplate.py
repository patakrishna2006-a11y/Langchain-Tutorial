from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = 'llama3.2:1b',
    temperature = 0
)

chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} Expert.'),
    ('human', 'Tell me about {topic}.')
])

prompt = chat_prompt.invoke({'domain': 'AI', 'topic': 'LangChain'})
result = llm.invoke(prompt)
print(result.text)