import joblib
import torch
import pickle
import os
from flask import Flask, request, jsonify

import src.utils.setup_nltk
from src.data.preprocess import clean_text
from src.models.att_lstm import AttentionLSTM
from src.utils.padding import pad_sequences
from src.utils.config import *

app = Flask(__name__)

# -------------------------
# Load SVM model
# -------------------------
svm_model = None
tfidf = None

if os.path.exists("models/svm.pkl") and os.path.exists("models/tfidf.pkl"):
    svm_model = joblib.load("models/svm.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    print("✅ SVM model loaded")
else:
    print("⚠️ SVM model not found")

# -------------------------
# Load Attention-LSTM
# -------------------------
dl_model = None
dl_tokenizer = None

if os.path.exists("models/att_lstm.pth") and os.path.exists("models/tokenizer.pkl"):
    with open("models/tokenizer.pkl", "rb") as f:
        dl_tokenizer = pickle.load(f)

    dl_model = AttentionLSTM(VOCAB_SIZE).to(DEVICE)
    dl_model.load_state_dict(
        torch.load("models/att_lstm.pth", map_location=DEVICE)
    )
    dl_model.eval()
    print("✅ Attention-LSTM loaded")
else:
    print("⚠️ Attention-LSTM not found")

# -------------------------
# API endpoint
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "endpoints": ["/predict"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}

    review = data.get("review", "")
    model_type = data.get("model", "svm")

    if not review.strip():
        return jsonify({"error": "Review text required"}), 400

    cleaned = clean_text(review)

    # ---------- SVM ----------
    if model_type == "svm":
        if svm_model is None or tfidf is None:
            return jsonify({"error": "SVM model not loaded"}), 500

        vec = tfidf.transform([cleaned])
        pred = svm_model.predict(vec)[0]

        return jsonify({
            "model": "svm",
            "sentiment": "positive" if pred == 1 else "negative"
        })

    # ---------- Attention-LSTM ----------
    elif model_type == "att_lstm":
        if dl_model is None or dl_tokenizer is None:
            return jsonify({"error": "Attention-LSTM not loaded"}), 500

        seq = dl_tokenizer.encode(cleaned)
        seq = pad_sequences([seq], MAX_LEN).to(DEVICE)

        with torch.no_grad():
            logit = dl_model(seq).squeeze()
            prob = torch.sigmoid(logit).item()

        return jsonify({
            "model": "attention_lstm",
            "sentiment": "positive" if prob >= 0.5 else "negative",
            "confidence": round(prob, 3)
        })

    else:
        return jsonify({
            "error": 'Invalid model type. Use "svm" or "att_lstm".'
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
