from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import re

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "kannada_emotion_model_xlm")

# Confidence threshold is only for info / UI message
CONFIDENCE_THRESHOLD = 0.60   # 60%

# MUST match the label mapping used in training.py
ID2LABEL = {
    0: "joy",
    1: "anger",
    2: "sadness",
    3: "fear",
    4: "neutral",
}

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("=" * 60)
print("LOADING KANNADA EMOTION CLASSIFIER (PyTorch)")
print("=" * 60)
# Try to load local model first. If model.safetensors is missing (e.g. on GitHub), fall back to Hugging Face Hub.
HF_MODEL_ID = "premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa" # <-- Replace with your actual Hugging Face ID!

local_weights_path = os.path.join(MODEL_DIR, "model.safetensors")
if os.path.exists(MODEL_DIR) and os.path.exists(local_weights_path):
    print(f"Loading model locally from: {MODEL_DIR}")
    model_path = MODEL_DIR
else:
    print(f"Local model weights not found. Loading from Hugging Face Hub: {HF_MODEL_ID}")
    model_path = HF_MODEL_ID

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"[OK] Model loaded. Using device: {device}")
print("=" * 60)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_input_text(text: str) -> str:
    """Light cleaning: normalize spaces, remove ZWJ etc."""
    text = text.replace("\u200c", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_sentence(sentence: str):
    """
    Run the model and return a dict in the same format
    your frontend expects (including 'success' key).
    """
    cleaned = clean_input_text(sentence)

    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # shape [num_labels]

    # Top prediction
    top_idx = int(probs.argmax())
    top_label = ID2LABEL[top_idx]          # e.g. "joy"
    top_score = float(probs[top_idx])      # 0–1

    # Build top-3 predictions list
    indices_sorted = probs.argsort()[::-1][:3]
    top3 = []
    for i in indices_sorted:
        lbl = ID2LABEL[int(i)]
        score = float(probs[int(i)])
        top3.append({
            "emotion": lbl,                    # e.g. "joy"
            "confidence": f"{score:.2%}",      # "97.53%"
            "score": score,                    # 0.9753
        })

    low_conf = top_score < CONFIDENCE_THRESHOLD

    # IMPORTANT: include 'success': True so frontend doesn't show "Server error"
    response = {
        "success": True,
        "original_sentence": sentence,
        "cleaned_sentence": cleaned,
        "predicted_emotion": top_label,           # "joy"
        "confidence": f"{top_score:.2%}",         # "81.00%"
        "confidence_raw": round(top_score, 4),    # 0.81
        "top_3_predictions": top3,
        "low_confidence": low_conf,
        "threshold": f"{CONFIDENCE_THRESHOLD:.0%}",  # "60%"
        "message": "Low confidence (threshold 60%)" if low_conf else "Prediction successful",
    }

    return response


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives JSON from frontend and returns prediction.
    Frontend sends: { "text": "<sentence>" }
    (older versions sometimes used "sentence", so we accept both.)
    """
    try:
        data = request.get_json(force=True) or {}
        text = (data.get("text") or data.get("sentence") or "").strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided."
            }), 400

        result = classify_sentence(text)
        return jsonify(result)

    except Exception as e:
        print("❌ Error during prediction:", e)
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Server error during prediction."
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_dir": MODEL_DIR,
        "device": str(device),
    })


if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
