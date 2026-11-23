from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough


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

joke_chain = RunnableSequence(joke_prompt, llm, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'Explanation' : RunnableSequence(summary_prompt, llm, parser)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)
result = final_chain.invoke({'topics' : 'AI'})
print("== Joke ==", "\n", result['joke'], "\n")
print("== Explanation ==", "\n", result['Explanation'], "\n")
final_chain.get_graph().print_ascii()