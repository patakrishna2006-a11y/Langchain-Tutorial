from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city and brief journey of a fictional person \n {name} return all the fields: name, age, city, journey (brief description), \n {format_instructions}",
    input_variables = ["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)   

chain =  template | llm | parser

final_result = chain.invoke({"name" : "Ash Ketchum"})
print(final_result)