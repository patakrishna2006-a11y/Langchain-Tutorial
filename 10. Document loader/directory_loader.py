
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"C:\Users\patak\OneDrive\Desktop\Model",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

document = loader.lazy_load()

for docs in document:
    print(docs.metadata)
