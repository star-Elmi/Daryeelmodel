import pickle
import torch
from sentence_transformers import SentenceTransformer, util
import openai
import os

# 🔐 Set your API key in system or VS Code terminal first:
# Windows:  set OPENROUTER_API_KEY=your_key_here
# Linux/macOS:  export OPENROUTER_API_KEY=your_key_here

# Load trained data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# Load sentence-transformers model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# AI fallback key
openai.api_key = os.getenv("OPENROUTER_API_KEY")

def ask_ai_fallback(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Fallback Error: {e}"

def get_answer(user_question, threshold=0.65):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, embeddings)[0]
    top_result_idx = torch.argmax(cosine_scores).item()
    top_score = cosine_scores[top_result_idx].item()

    if top_score >= threshold:
        return answers[top_result_idx] + f" (score: {top_score:.2f})"
    else:
        return ask_ai_fallback(user_question)

# Chat loop
print("🤖 Chatbot ready! Type 'exit' to quit.\n")
while True:
    user_input = input("🧑 You: ")
    if user_input.lower() == "exit":
        break
    reply = get_answer(user_input)
    print("🤖 Bot:", reply)
