from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=1
)

result = llm.invoke("write a 20 line poem about the love of indian teenage in a simple english.")
print(result.text)