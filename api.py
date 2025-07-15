from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('pa.model.pkl')  # ama model kale

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data['question']
    prediction = model.predict([question])
    return jsonify({'answer': prediction[0]})

if __name__ == '__main__':
    app.run(port=8000)
