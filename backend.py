# EcoSakhi: Gen AI Enhanced Backend
# Author: Shalini S K | Green Internship | 1M1B Cohort – 2025

from flask import Flask, request, jsonify
from transformers import pipeline
import random
import json
import datetime
import os

app = Flask(__name__)

# ------------------------- Load QA and Text Gen Pipelines -------------------------
qa_pipeline = pipeline("question-answering")
gen_pipeline = pipeline("text-generation", model="gpt2")

# ------------------------- Knowledge Base -------------------------
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        KNOWLEDGE = json.load(f)
except FileNotFoundError:
    KNOWLEDGE = {
        "faqs_context": "Renewable energy includes solar, wind, and hydro power.",
        "energy_tips": []
    }

# ------------------------- Sample Quiz Questions -------------------------
QUIZ_QUESTIONS = [
    {"question": "Which of the following is a renewable energy source?", "options": ["Coal", "Wind", "Oil", "Natural Gas"], "answer": "Wind"},
    {"question": "What device converts sunlight into electricity?", "options": ["Windmill", "Turbine", "Solar Panel", "Generator"], "answer": "Solar Panel"},
    {"question": "Which gas is primarily responsible for global warming?", "options": ["Oxygen", "Hydrogen", "Carbon Dioxide", "Nitrogen"], "answer": "Carbon Dioxide"}
]

# ------------------------- Localization -------------------------
SCHEMES_BY_STATE = {
    "karnataka": "Surya Raitha Scheme - Solar pump subsidy for farmers.",
    "tamil nadu": "Chief Minister's Solar Rooftop Capital Incentive Scheme.",
    "maharashtra": "MahaUrja Subsidy for Rooftop Solar Installations."
}

# ------------------------- Utility Functions -------------------------
def generate_energy_tip():
    prompt = "Suggest one practical energy-saving tip for Indian households."
    output = gen_pipeline(prompt, max_length=40, num_return_sequences=1, do_sample=True)
    return output[0]['generated_text'].replace(prompt, "").strip()

def get_scheme(state):
    return SCHEMES_BY_STATE.get(state.lower(), "No specific scheme found. Check your state renewable energy portal.")

def get_quiz():
    return random.choice(QUIZ_QUESTIONS)

# ------------------------- Chatbot Routes -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    state = request.json.get("state", "")

    if "quiz" in user_input.lower():
        return jsonify({"type": "quiz", "data": get_quiz()})
    elif "tip" in user_input.lower() or "save energy" in user_input.lower():
        return jsonify({"type": "tip", "data": generate_energy_tip()})
    elif "scheme" in user_input.lower() or "state" in user_input.lower():
        return jsonify({"type": "scheme", "data": get_scheme(state)})
    else:
        context = KNOWLEDGE.get("faqs_context", "Renewable energy includes solar, wind, and hydro sources.")
        response = qa_pipeline(question=user_input, context=context)
        return jsonify({"type": "answer", "data": response['answer'], "confidence": round(response['score'], 2)})

# ------------------------- Weekly Tip Endpoint -------------------------
@app.route("/weekly-tip", methods=["GET"])
def weekly_tip():
    tip = generate_energy_tip()
    date = datetime.date.today().isoformat()
    return jsonify({"date": date, "weekly_tip": tip})

# ------------------------- Launch -------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")

