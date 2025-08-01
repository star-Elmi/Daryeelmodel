# ✅ Full Somali Health Chatbot API with local + OpenAI fallback

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# ✅ Load environment variables
load_dotenv()

# ✅ Connect to OpenAI or OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ✅ Load trained data
with open("trained_data.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = data["embeddings"]

# ✅ Initialize embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI()

class Question(BaseModel):
    question: str

# ✅ Keyword Dictionaries
health_keywords = ["xanuun", "qandho", "neef", "lalabo", "calool", "madax", "wareer", "qufac",
                   "san", "sanka", "indho", "dhiig", "jug", "cudur", "hargab", "infekshan", "dareemayaa",
                   "xiiq", "matag", "xasaasiyad", "sonkor", "wadne", "karo", "kansar", "kaadi", "kaadida", "dhiigkarka"]

irrelevant_keywords = ["ciyaar", "film", "jacayl", "hees", "tiktok", "fanka", "ronaldo", "football", "musalsal", "aroos"]

emotion_keywords = ["cabsi", "cabsida", "walwal", "walbahaarka", "naxdin", "naxay", "welwel", "argagax", "dareen", "stress"]

definition_keywords = ["waa maxay", "macnaheedu waa", "micnaha", "sidee u faafa", "sidee u shaqeeyaa", "tilmaamaha", "tusaale", "noocee ah", "calaamado", "digniin"]

# ✅ Helper Functions
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def count_keywords(text: str, keywords: list) -> int:
    return sum(1 for word in keywords if word in text)

def is_health_related(text: str) -> bool:
    cleaned = clean_text(text)
    return count_keywords(cleaned, health_keywords) >= 1

def contains_emotion_trigger(text: str) -> bool:
    cleaned = clean_text(text)
    return count_keywords(cleaned, emotion_keywords) >= 1

def is_definition_question(text: str) -> bool:
    cleaned = clean_text(text)
    return any(phrase in cleaned for phrase in definition_keywords)

def is_recurrent(text: str) -> bool:
    return any(word in text for word in ["soo noqnoqda", "mar walba", "markasta", "markii hore"])

# ✅ Main Chat Endpoint
@app.post("/chat")
def chat(payload: Question):
    user_q = payload.question.strip()
    cleaned_q = clean_text(user_q)

    user_embed = model.encode(user_q, convert_to_tensor=True)
    scores = util.cos_sim(user_embed, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    if count_keywords(cleaned_q, irrelevant_keywords) > 0 and not is_health_related(cleaned_q):
        return {
            "source": "local",
            "answer": "❌ Su’aashan ma lahan xiriir caafimaad. Fadlan weydii su’aal caafimaad ah.",
            "score": round(best_score, 2)
        }

    if is_health_related(cleaned_q) and is_definition_question(cleaned_q):
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
                "answer": f"✅ Su’aashan waxay la xiriirtaa caafimaad. Jawaabta: {reply}",
                "score": round(best_score, 2)
            }
        except Exception as e:
            return {
                "source": "error",
                "answer": f"❌ Khalad ayaa dhacay marka la isticmaalayo OpenAI: {str(e)}",
                "score": 0.0
            }

    if is_health_related(cleaned_q) and contains_emotion_trigger(cleaned_q):
        try:
            completion = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful Somali health assistant. Answer shortly in Somali."},
                    {"role": "user", "content": user_q}
                ]
            )
            reply = completion.choices[0].message.content.strip()
            note = "💡 Calaamadahan waxa sababi kara falcelin shucuur sida cabsi ama walwal. ➕ Haddii xaaladdaadu ay sii socoto ama soo noqnoqoto, la xiriir dhakhtar."
            return {
                "source": "openai",
                "answer": f"✅ Su’aashan waxay la xiriirtaa caafimaad. Jawaabta: {reply} {note}",
                "score": round(best_score, 2)
            }
        except Exception as e:
            return {
                "source": "error",
                "answer": f"❌ Khalad ayaa dhacay marka la isticmaalayo OpenAI: {str(e)}",
                "score": 0.0
            }

    if is_health_related(cleaned_q) and best_score >= 0.70:
        response = answers[best_idx]
        extra = "➕ Haddii xaaladdaadu ay sii socoto, la xiriir dhakhtar."
        if is_recurrent(cleaned_q):
            extra = "➕ Haddii xaaladdaadu ay soo noqnoqoto, la xiriir dhakhtar."
        return {
            "source": "dataset",
            "answer": f"✅ Su’aashan waxay la xiriirtaa caafimaad. Jawaabta: {response} {extra}",
            "score": round(best_score, 2)
        }

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
            extra = "➕ Haddii xaaladdaadu ay sii socoto, la xiriir dhakhtar."
            if is_recurrent(cleaned_q):
                extra = "➕ Haddii xaaladdaadu ay soo noqnoqoto, la xiriir dhakhtar."
            return {
                "source": "openai",
                "answer": f"✅ Su’aashan waxay la xiriirtaa caafimaad. Jawaabta: {reply} {extra}",
                "score": round(best_score, 2)
            }
        except Exception as e:
            return {
                "source": "error",
                "answer": f"❌ Khalad ayaa dhacay marka la isticmaalayo OpenAI: {str(e)}",
                "score": 0.0
            }

    return {
        "source": "uncertain",
        "answer": "Ma hubo su’ashan. Fadlan dib u qor si cad ama i weydii su’aal caafimaad ah.",
        "score": round(best_score, 2)
    }

@app.get("/")
def root():
    return {"message": "✅ Somali Health Chatbot API is running. Use POST /chat to ask questions."}
