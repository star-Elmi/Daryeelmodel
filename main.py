# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# OpenRouter (Kimi AI) client
client = OpenAI(
    api_key=os.getenv("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"),
    base_url="https://openrouter.ai/api/v1"
)

# Load trained Q&A data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI()

# Request schema
class Question(BaseModel):
    question: str

# Check if the question is complex or low confidence
def is_complex_question(q: str, score: float) -> bool:
    q_lower = q.lower()
    return (
        len(q.split()) > 8 or
        "," in q or
        "iyo" in q_lower or
        "sharax" in q_lower or
        "sabab" in q_lower or
        score < 0.70
    )

# AI fallback with health filtering instruction
def ask_ai_fallback(question: str, top_examples: list = None):
    context = ""
    if top_examples:
        context = "\n".join([f"- {ex}" for ex in top_examples])

    prompt = f"""
You are a kind and helpful Somali medical assistant.

Please follow these rules carefully:

1. If the user sends a friendly greeting like "asc", "ma fiicantahay", or "hello", respond politely and say:  
   "👋 Waad salaamantahay! Waxaan ahay caawiye caafimaad. Weydii su'aal la xiriirta caafimaadka fadlan."

2. If the user's question is NOT related to health, do not answer it. Instead, respond with:  
   "❌ Su'aashan ma quseyso caafimaadka. Fadlan i weydii wax ku saabsan caafimaadka kaliya."

3. If the question is about health, use the context below to answer clearly and helpfully.

📘 Medical Context:
{context}

💬 User Question:
{question}
    """

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" AI fallback error: {e}"

# Main /chat endpoint
@app.post("/chat")
def get_answer(payload: Question):
    user_question = payload.question.strip()
    user_embed = model.encode(user_question, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embed, embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_match_idx].item()

    if not is_complex_question(user_question, best_score) and best_score >= 0.65:
        return {
            "source": "dataset",
            "answer": answers[best_match_idx],
            "score": round(best_score, 2)
        }
    else:
        top_indices = torch.topk(cosine_scores, 3).indices.tolist()
        top_context = [f"{questions[i]} — {answers[i]}" for i in top_indices]

        fallback = ask_ai_fallback(user_question, top_context)
        return {
            "source": "ai",
            "answer": fallback,
            "score": round(best_score, 2)
        }

# Health check
@app.get("/")
def root():
    return {"message": "AI API working"}