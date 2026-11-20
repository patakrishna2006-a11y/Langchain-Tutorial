from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

class person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The person's city")
    journey: str = Field(description="The person's journey")

parser = PydanticOutputParser(pydantic_object=person)

prompt = PromptTemplate(
    template="Give me a person with name, age, city, and brief description of journey of a fictional {name}\n {format_instructions}",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

final_result = chain.invoke({"name": "Ash Ketchum"})
print(final_result)