from langchain_nvidia_ai_endpoints import ChatNVIDIA
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = """Classify the sentiment of this feedback.

    Return exactly ONE word:
    Positive or Negative.

    Do not give explanations.

    Feedback: {feedback}""",
    input_variables=['feedback']
)

classifier_chain = prompt1 | llm | parser

prompt2 = PromptTemplate(
    template="Thank the customer for their positive feedback in a friendly and professional way. generate only one message, not more than one message. \nFeedback: {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Apologize for the customer's bad experience, acknowledge the issue, and offer helpful assistance professionally. generate only one message, not more than one message. \nFeedback: {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x == 'Positive', prompt2 | llm | parser),
    (lambda x:x == 'Negative', prompt3 | llm | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

final_result = chain.invoke({"feedback" : "This smartphone completely exceeds expectations with its blazing-fast performance and incredibly smooth user experience. The vibrant display makes streaming and gaming an absolute pleasure, while the exceptional battery life easily powers through a full day of heavy use without needing a recharge. Additionally, the camera quality is outstanding, capturing stunningly sharp and detailed photos even in low-light conditions. With its sleek, premium design and intuitive interface, this device perfectly combines style and functionality, making it an absolute joy to use every single day."})

print(final_result)
chain.get_graph().print_ascii()