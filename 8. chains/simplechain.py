from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

prompt = PromptTemplate(
    template = """Write 5 interesting facts about the topic {topic} in a simple english.""",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

final_result = chain.invoke({"topic" : "Pokemon"})
print(final_result)
chain.get_graph().print_ascii()