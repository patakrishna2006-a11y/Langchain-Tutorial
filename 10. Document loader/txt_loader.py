from langchain_community.document_loaders import TextLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

loader = TextLoader(r"C:\Users\patak\OneDrive\Desktop\Langchain\10. Document loader\Cricket.txt")
docs = loader.load()

print(docs[0].page_content, "\n")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary for the following text. {text}",
    input_variables=['text']
)

chain = prompt | llm | parser

final_result = chain.invoke({'text' : docs[0].page_content})
print(final_result)