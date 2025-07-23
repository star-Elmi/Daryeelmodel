import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# OpenRouter Moonshot client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Load trained data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ask_ai_fallback(question):
    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",  # ✅ Free Moonshot model
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
    print(f"🤖 Bot: {reply}\n")