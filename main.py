from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import openai
import pickle
import os
from dotenv import load_dotenv

# Load environment variables (API KEY)
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")

# Load trained data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# Load sentence-transformers model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FastAPI instance
app = FastAPI()

class Question(BaseModel):
    question: str

def ask_ai_fallback(q: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": q}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI fallback error: {e}"

@app.post("/chat")
def get_answer(payload: Question):
    user_question = payload.question
    user_embed = model.encode(user_question, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embed, embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_match_idx].item()

    if best_score >= 0.65:
        return {
            "source": "dataset",
            "answer": answers[best_match_idx],
            "score": round(best_score, 2)
        }
    else:
        fallback = ask_ai_fallback(user_question)
        return {
            "source": "ai",
            "answer": fallback,
            "score": round(best_score, 2)
        }
