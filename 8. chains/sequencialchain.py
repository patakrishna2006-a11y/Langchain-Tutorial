from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

prompt1 = PromptTemplate(
    template = """Write 5 interesting facts about the topic {topic} in a simple english.""",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Write a summary of the following text in simple english: {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

final_result = chain.invoke({"topic" : "Pokemon"})
print(final_result)
chain.get_graph().print_ascii()