# Embedding single queries
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

result = embeddings.embed_query("New Delhi is the capital of India.")

print(result)
print("Embedding dimensions:", len(result))

# Embedding multiple queries
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

queries = [
    "New Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan."
]

results = embeddings.embed_documents(queries)

for i, result in enumerate(results):
    print(f"Query {i+1}: {result}")
    print(f"Embedding dimensions: {len(result)}")