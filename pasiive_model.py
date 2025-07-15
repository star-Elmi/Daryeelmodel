# Passive Aggressive Classifier - Cleaned, Balanced & Tuned for Accuracy with Comments

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load and read Excel data
file_path = "full_dataset_cleaned_suaalo_jawaabo.xlsx"
df = pd.read_excel(file_path)

# Remove rows with missing labels
df = df.dropna(subset=['Label'])

# Normalize labels (strip spaces and lower case)
df['Label'] = df['Label'].str.strip().str.lower()

# Merge similar labels to reduce noise and confusion
label_map = {
    'blood tests': 'blood test',
    'follow-up': 'follow_up',
    'complication': 'complications',
    'when to see doctor': 'when_to_see_doctor',
    'when to see a doctor': 'when_to_see_doctor',
    'causes': 'cause',
    'symptoms in infants': 'symptoms',
    'diagnostic tests': 'tests',
    'types': 'classification',
    'general info': 'overview',
    'other': 'overview'
}
df['Label'] = df['Label'].replace(label_map)

# Remove labels with fewer than 3 samples (model can't learn from too few examples)
label_counts = df['Label'].value_counts()
valid_labels = label_counts[label_counts >= 3].index
df = df[df['Label'].isin(valid_labels)]

# Balance dataset using oversampling to equalize class sizes
max_size = df['Label'].value_counts().max()
df_balanced = df.groupby('Label').apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)

# Combine "Su'aal" and "Jawaab" for input features
X = df_balanced["Suaal"] + " " + df_balanced["Jawaab"]
y = df_balanced['Label']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model using TF-IDF + PassiveAggressiveClassifier
# TF-IDF is tuned with better settings: bigrams, stopwords, sublinear TF

def train_model(X_train, y_train, X_test, y_test, model_path='pa_model.pkl'):
    pipeline = make_pipeline(
        TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=15000,
            min_df=3,
            max_df=0.85
        ),
        PassiveAggressiveClassifier(random_state=42, max_iter=1000)
    )
    pipeline.fit(X_train, y_train)  # Train model
    y_pred = pipeline.predict(X_test)  # Predict test set
    accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
    report = classification_report(y_test, y_pred)  # Full classification report
    joblib.dump(pipeline, model_path)  # Save model to file
    print(f"Model saved to {model_path}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

# Prediction function: load model and predict from new text input
def predict_label(text, model_path='pa_model.pkl'):
    model = joblib.load(model_path)
    prediction = model.predict([text])[0]
    return prediction

# Run model training and predict example
if __name__ == '__main__':
    train_model(X_train, y_train, X_test, y_test)
    new_question = "Maxay yihiin calaamadaha cudurka Zika?"
    print("Prediction:", predict_label(new_question))
