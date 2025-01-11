import streamlit as st
import pickle
import random

# Load the saved model and vectorizer
with open('chatbot_model1.pkl', 'rb') as model_file:
    vectorizer, clf, intents = pickle.load(model_file)

# Function to generate a chatbot response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    # Page Title
    st.title("🤖 Chatbot Assistant")
    st.markdown(
        """
        Welcome to your AI assistant chatbot! Feel free to ask me anything.  
        Here's a list of sample questions to get you started. 🧠💬
        """
    )

    # Divider
    st.markdown("---")

    # Example Questions Section
    st.subheader("📋 Example Questions You Can Ask")
    with st.expander("Click to see example questions 👇"):
        for intent in intents:
            st.markdown(f"### **{intent['tag'].capitalize()}**")
            for pattern in intent['patterns']:
                st.markdown(f"- {pattern}")

    # Divider
    st.markdown("---")

    # Chat Section
    st.subheader("💬 Chat with Me!")
    user_input = st.text_input("Type your message here 👇")

    if user_input:
        response = chatbot(user_input)
        st.markdown(f"### 🤖 Chatbot Response: \n {response}")

        if response.lower() in ['goodbye', 'bye']:
            st.markdown("### 👋 Thank you for chatting with me! Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
