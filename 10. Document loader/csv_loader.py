from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r"C:\Users\patak\OneDrive\Desktop\Langchain\10. Document loader\cricket.csv")

docs = loader.load()
print(len(docs))
print(docs[0])
