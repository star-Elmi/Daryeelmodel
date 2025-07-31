from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

# ✅ Load environment variables
load_dotenv()

# ✅ OpenRouter AI client (optional if fallback is still needed in the future)
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ✅ Load trained data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# ✅ Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ FastAPI app
app = FastAPI()

# ✅ Request schema
class Question(BaseModel):
    question: str

# ✅ Detect complex/irrelevant question
def is_irrelevant_question(q: str, score: float) -> bool:
    q_lower = q.lower()
    keywords = ["ciyaar", "romantic", "jacayl", "hees", "politics", "film", "football", "cristiano", "ronaldo", "barnaamij", "tiktok", "fanka"]
    return any(k in q_lower for k in keywords) or score < 0.65

# ✅ POST /chat endpoint
@app.post("/chat")
def get_answer(payload: Question):
    user_question = payload.question.strip()
    user_embed = model.encode(user_question, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embed, embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_match_idx].item() # type: ignore

    # ✅ If question is irrelevant, return local message without calling AI
    if is_irrelevant_question(user_question, best_score):
        return {
            "source": "local",
            "answer": "❌ Su'aashan ma quseyso caafimaadka. Fadlan i weydii wax ku saabsan caafimaadka kaliya.",
            "score": round(best_score, 2)
        }

    # ✅ Otherwise, respond with closest match from dataset
    return {
        "source": "dataset",
        "answer": answers[best_match_idx],
        "score": round(best_score, 2)
    }

# ✅ Root endpoint
@app.get("/")
def root():
    return {"message": "✅ Somali Health Chatbot API is running. Use POST /chat"}
