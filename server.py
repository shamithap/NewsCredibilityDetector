#server.py
from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.load_weights('./model/final_model.weights.h5')  # Load the saved weights
print("Model and weights loaded successfully.")

@app.route('/check', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Preprocess the input text
    encodings = tokenizer([text], return_tensors='tf', truncation=True, padding=True, max_length=100)
    preds = model(encodings)[0]
    prediction = tf.argmax(preds, axis=1).numpy()
    is_fake = bool(prediction[0] == 1)  # True for fake news, False for real news
    
    return jsonify({"is_fake": is_fake})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
