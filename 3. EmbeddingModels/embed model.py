# Embed single query
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import dotenv

dotenv.load_dotenv()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-8b"
)

result = embeddings.embed_query("New Delhi is the capital of India.")

print(result)
print("Embedding dimensions:", len(result))

# Embed multiple queries
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import dotenv

dotenv.load_dotenv()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-8b"
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

