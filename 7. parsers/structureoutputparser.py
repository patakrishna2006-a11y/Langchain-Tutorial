from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

schema = [
    ResponseSchema(name="fact_1", description="fact 1 about the topics"),
    ResponseSchema(name="fact_2", description="fact 2 about the topics"),
    ResponseSchema(name="fact_3", description="fact 3 about the topics")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me 3 facts about the topics {topics} \n {format_instructions}",
    input_variables = ["topics"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | llm | parser

final_result = chain.invoke({"topics": "Black holes"})

print(final_result)