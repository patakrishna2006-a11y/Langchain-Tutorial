from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel


dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

parser = StrOutputParser()

X_prompt = PromptTemplate(
    template="Generate the one single tweek about {topics}",
    input_variables=['topics']
)

linkdein_prompt = PromptTemplate(
    template="Generate the one single Linkdein post about {topics}",
    input_variables=['topics']
)

parallel_chain = RunnableParallel({
    'tweek' : RunnableSequence(X_prompt, llm, parser),
    'linkdein_post' : RunnableSequence(linkdein_prompt, llm, parser)
})

result = parallel_chain.invoke({'topics' : "AI"})
print("== tweek ==", "\n", result['tweek'])
print("== linkdein post ==", "\n", result['linkdein_post'])
parallel_chain.get_graph().print_ascii()