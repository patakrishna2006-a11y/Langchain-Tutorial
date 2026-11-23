from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough


dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Write a complete report on the topics {topics}",
    input_variables = ['topics']
)

prompt2 = PromptTemplate(
    template = 'Generate the summary of the report in 300 words. {text}',
    input_variables= ['text']
)

report_chain = RunnableSequence(prompt, llm, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_chain, branch_chain)

final_result = final_chain.invoke({'topics' : 'AI'})
print(final_result)
print("END")