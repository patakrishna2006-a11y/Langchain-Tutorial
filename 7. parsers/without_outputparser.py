from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate

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

report_prompt = report_template.invoke({"topics": "Black Hole"})

report_result = llm.invoke(report_prompt)

print("Report ")
print(report_result.content, "\n")

summary_prompt = summary_template.invoke({"text": report_result})

summary_result = llm.invoke(summary_prompt)

print("Summary ")
print(summary_result.content)