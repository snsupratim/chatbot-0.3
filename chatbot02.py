import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import nltk

# Download NLTK data
nltk.download('punkt')

# Load the saved model, vectorizer, and intents
model = load_model('chatbot_model.h5')  # Neural network model
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('intents.pkl', 'rb') as file:
    intents = pickle.load(file)

# Function to get chatbot response
def chatbot_response(input_text):
    input_vector = vectorizer.transform([input_text]).toarray()
    prediction = model.predict(input_vector)
    tag_index = np.argmax(prediction)
    predicted_tag = list(set(intent['tag'] for intent in intents))[tag_index]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Chatbot",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for better design
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput > div > div {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextArea > div {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 10px;
        }
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and description
    st.title("💬 AI-Powered Chatbot")
    st.write("Welcome to the chatbot! Type a message to interact and get responses. 🤖")

    # Sidebar examples
    st.sidebar.title("🤔 Example Questions")
    st.sidebar.write("Here are some example questions you can ask:")
    for intent in intents:
        st.sidebar.write(f"- {random.choice(intent['patterns'])}")

    # User input
    user_input = st.text_input(
        "You:",
        placeholder="Type your message here...",
        help="Start a conversation by typing something.",
        key="user_input"
    )

    # Chatbot response area
    if user_input:
        response = chatbot_response(user_input)
        st.text_area(
            "Chatbot:",
            value=response,
            height=100,
            max_chars=None,
            key="response",
            help="The chatbot's response will appear here."
        )

        if response.lower() in ['goodbye', 'bye']:
            st.success("Thank you for chatting! Have a great day! 🌟")

if __name__ == "__main__":
    main()
