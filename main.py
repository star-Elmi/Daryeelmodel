from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# ✅ Load .env environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("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"),  # Ku qor .env file: OPENROUTER_API_KEY=sk-xxxxx
    base_url="https://openrouter.ai/api/v1"
)

# ✅ Load dataset
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# ✅ Embedder
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI()

class Question(BaseModel):
    question: str

# ✅ Keywords
health_keywords = [
    "xanuun", "qandho", "neef", "lalabo", "calool", "madax", "wareer", "qufac",
    "san", "sanka", "indho", "dhiig", "jug", "cudur", "hargab", "infekshan", "dareemayaa",
    "xiiq", "matag", "xasaasiyad", "sonkor", "wadne", "karo", "kansar", "kaadi", "kaadida", "dhiigkarka"
]

irrelevant_keywords = [
    "ciyaar", "film", "jacayl", "hees", "tiktok", "fanka", "ronaldo", "football", "musalsal", "aroos"
]

# ✅ Clean & count helper
def clean_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def count_keywords(text: str, keywords: list) -> int:
    return sum(1 for word in keywords if word in text)

def is_health_related(text: str) -> bool:
    cleaned = clean_text(text)
    return count_keywords(cleaned, health_keywords) >= 1  # at least 1 health keyword

@app.post("/chat")
def chat(payload: Question):
    user_q = payload.question.strip()
    cleaned_q = clean_text(user_q)

    # 1. Embed + similarity
    user_embed = model.encode(user_q, convert_to_tensor=True)
    scores = util.cos_sim(user_embed, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    # 2. Irrelevant only → reject
    if count_keywords(cleaned_q, irrelevant_keywords) > 0 and not is_health_related(cleaned_q):
        return {
            "source": "local",
            "answer": "❌ Su’aashan ma lahan xiriir caafimaad. Fadlan wax caafimaad ah i weydii.",
            "score": round(best_score, 2)
        }

    # 3. Health-related + confident score
    if is_health_related(cleaned_q) and best_score >= 0.70:
        return {
            "source": "dataset",
            "answer": answers[best_idx],
            "score": round(best_score, 2)
        }

    # 4. Health-related + low score → Use OpenRouter
    if is_health_related(cleaned_q):
        try:
            completion = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful Somali health assistant. Answer shortly in Somali."},
                    {"role": "user", "content": user_q}
                ]
            )
            reply = completion.choices[0].message.content.strip()
            return {
                "source": "openai",
                "answer": reply,
                "score": round(best_score, 2)
            }
        except Exception as e:
            return {
                "source": "error",
                "answer": f"❌ Khalad ayaa dhacay marka la isticmaalayo OpenRouter API: {str(e)}",
                "score": 0.0
            }

    # 5. Unknown case
    return {
        "source": "uncertain",
        "answer": "Ma hubo su’aashan. Fadlan dib u qor su’aasha si cad ama i weydii su’aal caafimaad ah.",
        "score": round(best_score, 2)
    }

@app.get("/")
def root():
    return {"message": "✅ Somali Health Chatbot API is active"}
