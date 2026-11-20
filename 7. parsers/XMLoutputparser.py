from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import XMLOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

parser = XMLOutputParser()

template = PromptTemplate(
    template = """Give me 3 facts about the topics {topics}   Return the facts in XML format, using:
     <facts>
        <fact_1>...</fact_1>
        <fact_2>...</fact_2>
        <fact_3>...</fact_3>
     </facts>
     \n {format_instructions}""",
    input_variables = ["topics"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | llm | parser

final_result = chain.invoke({"topics": "Black holes"})

print(final_result)
print(type(final_result))