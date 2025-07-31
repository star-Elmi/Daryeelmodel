import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report
import torch
import pickle
import os

dataset_path = "full_dataset_cleaned_suaalo_jawaabo.xlsx"
output_path = "trained_data.pkl"

# Load dataset
df = pd.read_excel(dataset_path)
print("🧾 Columns:", df.columns.tolist())  # Show columns

# Use correct Somali column names
questions = df["Suaal"].astype(str).tolist()
answers = df["Jawaab"].astype(str).tolist()
labels = df["Label"].astype(str).tolist()

print(f"✅ Loaded {len(questions)} samples.")

# Encode questions
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(questions, convert_to_tensor=True)

# Save model data
with open(output_path, "wb") as f:
    pickle.dump({
        "questions": questions,
        "answers": answers,
        "labels": labels,
        "embeddings": embeddings
    }, f)

print("✅ Model trained and saved to:", output_path)

# Evaluate accuracy
print("📊 Running evaluation...")
preds = []
for i, q in enumerate(questions):
    q_embed = model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_embed, embeddings)[0]
    best_match = torch.argmax(scores).item()
    preds.append(labels[best_match]) # type: ignore

print("\n🎯 Classification Report:\n")
print(classification_report(labels, preds))
