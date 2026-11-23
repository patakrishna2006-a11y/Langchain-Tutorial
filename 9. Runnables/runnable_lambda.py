from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


dotenv.load_dotenv()

def word_count(joke):
    return len(joke.split())

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

joke_prompt = PromptTemplate(
    template = "Write a joke about {topics}",
    input_variables=['topics']
)

parser = StrOutputParser()

joke_chain = RunnableSequence(joke_prompt, llm, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)
result  = final_chain.invoke({'topics': 'AI'})
print("Joke:", result['joke'], "\n" "Word Count:", result['word_count'])
final_chain.get_graph().print_ascii()