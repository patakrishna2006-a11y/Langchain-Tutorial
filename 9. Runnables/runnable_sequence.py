from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

joke_prompt = PromptTemplate(
    template = "Write a joke about {topics}",
    input_variables=['topics']
)

parser = StrOutputParser()

summary_prompt = PromptTemplate(
    template = "Explain the joke in simple english {joke}",
    inpit_variable = ['joke']
)

chain = RunnableSequence(joke_prompt, llm, parser, summary_prompt, llm, parser)
result = chain.invoke({'topics' : 'AI'})
print(result)
chain.get_graph().print_ascii()