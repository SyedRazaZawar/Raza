import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
import requests
import json
import io
from PIL import Image

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Prompt Template for Chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ('user', "Question: {question}")
    ]
)

# Singleton instance for the model
llm_instance = None

def get_llm(api_key, engine):
    global llm_instance
    if llm_instance is None:
        llm_instance = ChatGoogleGenerativeAI(
            model=engine,
            google_api_key=api_key,
            allow_reuse=True  # Allow reuse to prevent duplicates
        )
    return llm_instance

def generate_response(question, api_key, engine, temperature, max_token):
    try:
        llm = get_llm(api_key, engine)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

# Function to call Hugging Face API for Text-to-Speech
def tts_query(payload):
    response = requests.post("https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits", 
                             headers={"Authorization": "Bearer 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"}, 
                             json=payload)
    return response.content

# Function to call Hugging Face API for Text-to-Image
def image_query(payload):
    response = requests.post("https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0", 
                             headers={"Authorization": "Bearer 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"}, 
                             json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# Function to call Hugging Face API for Text-to-Music
def music_query(payload):
    response = requests.post("https://api-inference.huggingface.co/models/facebook/musicgen-small", 
                             headers={"Authorization": "Bearer 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"}, 
                             json=payload)
    return response.content

# Streamlit application layout
st.title("AI Model Selector")
st.write("Select a model from the dropdown to perform various tasks.")

# Model selection
model_selection = st.selectbox("Select Model", 
                                 ["Q&A Chatbot", "Text-to-Speech", "Text-to-Image", "Text-to-Music"])

# Q&A Chatbot
if model_selection == "Q&A Chatbot":
    st.sidebar.title("Chatbot Settings")
    api_key = st.sidebar.text_input("Please enter your Gemini API Key", type='password')
    engine = st.sidebar.selectbox('Select Gemini Model', ['gemini-1.5-flash'])
    temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
    max_token = st.sidebar.slider('Max Token', min_value=100, max_value=300, value=150)
    
    st.write("Please ask your Question:")
    user_input = st.text_input("Your Prompt:")
    
    if user_input and api_key:
        answer = generate_response(user_input, api_key, engine, temperature, max_token)
        st.write("Response:", answer)

# Text-to-Speech
elif model_selection == "Text-to-Speech":
    st.write("Enter some text, and the AI will generate speech for you!")
    input_text = st.text_area("Enter Text", value="The answer to the universe is 42")
    
    if st.button("Generate Audio"):
        if input_text.strip():
            with st.spinner("Generating audio..."):
                audio_data = tts_query({"inputs": input_text})
                st.success("Audio generated successfully!")
                st.audio(audio_data, format="audio/wav")
        else:
            st.error("Please enter some text.")

# Text-to-Image
elif model_selection == "Text-to-Image":
    st.write("Enter a prompt to generate an image:")
    user_input = st.text_input("Your prompt", value="")
    
    if st.button("Generate Image"):
        if user_input:
            try:
                image_bytes = image_query({"inputs": user_input})
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt.")

# Text-to-Music
elif model_selection == "Text-to-Music":
    st.write("Enter a description of the music you want to generate:")
    user_input = st.text_input("Description", "liquid drum and bass, atmospheric synths, airy sounds")
    
    if st.button("Generate Music"):
        if user_input:
            with st.spinner("Generating audio..."):
                audio_bytes = music_query({"inputs": user_input})
                st.audio(audio_bytes, format="audio/wav")
        else:
            st.warning("Please enter a description.")

# To run this app, use streamlit run streamlit_app.py in your terminal.
