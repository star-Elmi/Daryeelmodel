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
    api_key=os.getenv("OPENAI_API_KEY"),
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

# Helper: check if the question is complex
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

# Enhanced fallback using Kimi AI + dataset context
def ask_ai_fallback(question: str, top_examples: list = None):
    context = ""
    if top_examples:
        context = "\n".join([f"- {ex}" for ex in top_examples])

    prompt = f"""
You are a helpful medical AI assistant. Here's some relevant information from a trusted dataset:

{context}

Now answer this user question as clearly and helpfully as possible:
{question}
    """

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI fallback error: {e}"

# Main chat endpoint
@app.post("/chat")
def get_answer(payload: Question):
    user_question = payload.question.strip()
    user_embed = model.encode(user_question, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embed, embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_match_idx].item()

    if not is_complex_question(user_question, best_score) and best_score >= 0.65:
        # Simple question → Dataset answer
        return {
            "source": "dataset",
            "answer": answers[best_match_idx],
            "score": round(best_score, 2)
        }
    else:
        # Complex question → AI fallback with dataset context
        top_indices = torch.topk(cosine_scores, 3).indices.tolist()
        top_context = [f"{questions[i]} — {answers[i]}" for i in top_indices]

        fallback = ask_ai_fallback(user_question, top_context)
        return {
            "source": "ai",
            "answer": fallback,
            "score": round(best_score, 2)
        }
