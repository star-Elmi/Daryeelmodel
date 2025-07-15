print("🚀 Script started!")  # halkan hore
print("✅ Model training started")  # Bilaabista tababarka

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import torch
import os
import sys

dataset_path = "full_dataset_cleaned_suaalo_jawaabo.xlsx"
output_path = "trained_data.pkl"

print("📄 Checking dataset file...")
print("📂 Current folder files:", os.listdir())  # Show existing files in folder

if not os.path.exists(dataset_path):
    print(f"❌ Dataset file '{dataset_path}' not found.")
    sys.exit()

try:
    print("🟡 Reading Excel file...")
    df = pd.read_excel(dataset_path)
    print("✅ Excel loaded")
    print(f"🔢 Rows loaded: {len(df)}")

    print("🧾 Columns in Excel:", df.columns.tolist())  # ✅ Show all columns

    print("📊 Extracting columns...")
    questions = df["suaal"].astype(str).tolist()      # ✅ UPDATED
    answers = df["jawaab"].astype(str).tolist()       # ✅ UPDATED
    labels = df["label"].astype(str).tolist()
    print("✅ Column extraction complete.")

    print("📦 Loading sentence-transformers model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded")

    print("💡 Encoding questions...")
    embeddings = model.encode(questions, convert_to_tensor=True)
    print("✅ Encoding complete.")

    print("💾 Saving trained data to file...")
    with open(output_path, "wb") as f:
        pickle.dump({
            "questions": questions,
            "answers": answers,
            "labels": labels,
            "embeddings": embeddings
        }, f)
    print(f"✅ Training complete. File saved as: {output_path}")

except Exception as e:
    print("❌ ERROR occurred:")
    print(e)
