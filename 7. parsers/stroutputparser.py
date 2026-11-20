from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

# Create a prompt template for the report
report_template = PromptTemplate(
    template = "Write a detailed report on {topics}.",
    input_variables = ["topics"]
)

# create a summary of the report
summary_template = PromptTemplate(
    template = "Write a 5 line summary on the following {text}. ",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain1 = report_template | llm | parser

report_result = chain1.invoke({"topics": "Black Hole"})

print("Report ")
print(report_result, "\n")

chain2 = summary_template | llm | parser
summary_result = chain2.invoke({"text": report_result})

print("Summary ")
print(summary_result)