import requests
from flask import Flask, render_template, request

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/chat"  # FastAPI endpoint

@app.route("/")
def home():
    return render_template("index.html")  # Render a simple input form

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    response = requests.post(FASTAPI_URL, json={"text": user_input})
    if response.status_code == 200:
        chatbot_response = response.json().get("response")
        return render_template("chat.html", user_input=user_input, response=chatbot_response)
    else:
        return "Error: Could not get response from the chatbot server."

if __name__ == "__main__":
    app.run(debug=True)
