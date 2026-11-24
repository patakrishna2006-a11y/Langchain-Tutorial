from langchain_community.document_loaders import WebBaseLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

url = 'https://www.amazon.in/gp/product/B0DGHZWBYB?aaxitk=&aref=&pd_rd_plhdr=t&pf_rd_r=CH9V4N7J0Q0QB79VEVJQ'
loader = WebBaseLoader(url)

parser = StrOutputParser()

docs = loader.load()

prompt = PromptTemplate(
    template="answer the following question {question} from the following text {text}",
    input_variables=['text']
)

chain = prompt | llm | parser

final_result = chain.invoke({'text' : docs[0].page_content, 'question' : "generate the full summary of all products including all feature and everythings."})

print(final_result)