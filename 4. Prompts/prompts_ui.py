from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import load_prompt
import streamlit as st
import dotenv
import os

dotenv.load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")

st.header("Poem Writer Chatbot")

user_input = st.text_input("Enter your prompt:")

style_input = st.selectbox("Select a style:", ["Poem", "Story", "Essay", "Joke", "Letter"])
tone_input = st.selectbox("Select a tone:", ["Funny", "Informal", "Humorous", "Serious", "Friendly"])

template = load_prompt("poem_chatbot_template.json")

if st.button("Submit"):
     # Input Validation
    if not user_input.strip():
        st.warning("Please enter your prompt.")

    elif not style_input or not tone_input:
        st.warning("Please select both style and tone.")

    elif not MODEL_NAME:
        st.error("MODEL_NAME is missing from your .env file.")

    else:
        try:
            # Initialize LLM using environment variable
            llm = ChatNVIDIA(
                model=MODEL_NAME,
                temperature=0
            )

            # Generate Prompt
            prompt = template.format(
                user_input=user_input.strip(),
                style_input=style_input,
                tone_input=tone_input
            )

            # Invoke LLM
            with st.spinner("Generating your response..."):
                result = llm.invoke(prompt)

            st.success("Response generated successfully!")
            st.write(result.text)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
