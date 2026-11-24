from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\patak\OneDrive\Desktop\Langchain\10. Document loader\LangChain Notes.pdf")
docs = loader.load()

print(docs)
print(len(docs))