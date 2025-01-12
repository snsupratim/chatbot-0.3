from fastapi import FastAPI
import pickle
import random
from pydantic import BaseModel

# Load the saved model and vectorizer
with open('chatbot_model1.pkl', 'rb') as model_file:
    vectorizer, clf, intents = pickle.load(model_file)

# Define a request model
class UserInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Create the chatbot response function
def chatbot_response(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return {"response": random.choice(intent['responses'])}
    return {"response": "Sorry, I didn't understand that."}

# Define the FastAPI route
@app.post("/chat")
def chat(input_data: UserInput):
    return chatbot_response(input_data.text)


