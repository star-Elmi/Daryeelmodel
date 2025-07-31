# --- Step 1: Imports ---
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
import re

# --- Step 2: Load Dataset & Model Once at Startup ---
print("📦 Loading dataset...")
df = pd.read_excel("full_dataset_cleaned_suaalo_jawaabo.xlsx")

questions = df["Suaal"].astype(str).tolist()
answers = df["Jawaab"].astype(str).tolist()
labels = df["Label"].astype(str).tolist()

print("🤖 Loading model...")
model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

print("🔍 Embedding questions...")
question_embeddings = model.encode(questioans)

# Build health-related vocabulary
print("📚 Building vocabulary...")
health_vocab = set()
for q in questions:
    tokens = re.findall(r'\b\w+\b', q.lower())
    health_vocab.update(tokens)
health_vocab = list(health_vocab)

# --- Step 3: Flask App Setup ---
app = Flask(__name__)

# --- Step 4: Analyze Input Function ---
def analyze_input(text, threshold=0.65):
    symptoms = ["qandho", "wareer", "madax xanuun"]
    if all(symptom in text.lower() for symptom in symptoms):
        return "✅ Waxaa laga yaabaa inaad qabto xanuun la xiriira qandho, wareer iyo madax xanuun. La xiriir dhakhtar."

    user_embed = model.encode(text)
    scores = util.cos_sim(user_embed, question_embeddings)[0]
    best_score = float(np.max(scores))
    best_idx = int(np.argmax(scores))

    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return "❌ Qoraalka lama fahmin."

    health_related = sum(1 for w in words if w in health_vocab)
    health_percent = (health_related / len(words)) * 100

    if best_score < threshold:
        return f"❌ Ma aha su’aal caafimaad la xiriirta (Sim score: {best_score:.2f})"

    if health_percent < 40:
        return f"⚠️ {health_percent:.1f}% qoraalkaagu caafimaad kuma filna."

    return f"✅ Jawaab: {answers[best_idx]}\n(Label: {labels[best_idx]})"

# --- Step 5: POST endpoint for chatbot ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "Fadlan dir fariin qoraal ah."}), 400

    response = analyze_input(user_input)
    return jsonify({"response": response})

# --- Step 6: Run App ---
if __name__ == '__main__':
    print("🚀 Starting Chatbot API on http://localhost:5000")
    app.run(debug=True)
