from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["user_input", "style_input", "tone_input"],
    template="""You are an expert poem writer. 

    Your task is to transform the core information provided below into a specific writing style and tone.

    Core Information to Process:
    {user_input}

    Required Output Style:
    {style_input}

    Required Tone:
    {tone_input}

    Execution Instructions:
    1. Rewrite the core information entirely in the requested style and tone.
    2. Maintain the accuracy of the facts provided in the user input.
    3. Do not add unnecessary meta-commentary or introduction; start directly with the output.
    """
)

template.save("poem_chatbot_template.yaml")
template.save("poem_chatbot_template.json")