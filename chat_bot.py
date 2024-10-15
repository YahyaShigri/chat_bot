from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable (or from Streamlit secrets if using that)
api_key = os.getenv("GOOGLE_API_KEY")  # Use st.secrets["GOOGLE_API_KEY"] for Streamlit Sharing

# Streamlit app configuration
st.set_page_config(page_title="AI Text Assistant", page_icon="🤖")
st.title('AI Chatbot')

# Greeting message
st.markdown("Hello! I'm your AI assistant. I can help answer questions about technology, education, and general knowledge. How can I assist you today?")
st.image("https://botnation.ai/site/wp-content/uploads/2024/01/chatbot-drupal.webp", use_column_width=True)

# Check if the user query is app-related (questions about the assistant's creation or developer)
def is_app_related_query(query):
    keywords = [
        "create", "develop", "make", "creator", "who", "author", "about you", 
        "built", "designed", "constructed", "programmed", 
        "engineered", "invented", "origin", "authored", 
        "who made", "who created", "who built", "who designed", 
        "who constructed", "who invented"
    ]
    return any(keyword in query.lower() for keyword in keywords) and "you" in query.lower()

# Create AI assistant prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant. Please respond to user queries in English, but understand the questions in English."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Initialize chat message history
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Set up AI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create chain to generate responses using the model and output parser
chain = prompt | model | StrOutputParser()

# Combine the chain with message history tracking
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# User input handling
user_input = st.text_input("Enter your question in English:", "")

if user_input:
    # Check if the query is related to the app/assistant's creation
    if is_app_related_query(user_input):
        st.chat_message("assistant").write("I was created by Yahya Khan Shigri. You can find him at www.linkedin.com/in/yahya-khan-shigri.")
    else:
        st.chat_message("human").write(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            config = {"configurable": {"session_id": "any"}}
            
            response = chain_with_history.stream({"question": user_input}, config)
            
            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

else:
    st.warning("Please enter your question.")
