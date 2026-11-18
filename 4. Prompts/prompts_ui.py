from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import load_prompt
import streamlit as st
import dotenv

dotenv.load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
)

st.header("Poem Writer Chatbot")

user_input = st.text_input("Enter your prompt:")

style_input = st.selectbox("Select a style:", ["Poem", "Story", "Essay", "Joke", "Letter"])
tone_input = st.selectbox("Select a tone:", ["Funny", "Informal", "Humorous", "Serious", "Friendly"])

template = load_prompt("poem_chatbot_template.json")

if st.button("Submit"):
    prompt = template.format(user_input=user_input, style_input=style_input, tone_input=tone_input)
    result = llm.invoke(prompt)
    st.write(result.text)