from flask import Flask, request, jsonify, render_template
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
# LOAD MODEL / API MODE SELECTION
# --------------------------------------------------

HF_MODEL_ID = "premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa"
local_weights_path = os.path.join(MODEL_DIR, "model.safetensors")

# Use API Mode if explicitly requested, or if local model weights are missing (e.g. cloud deployment)
USE_API_MODE = os.environ.get("USE_API_MODE", "false").lower() == "true"

if not (os.path.exists(MODEL_DIR) and os.path.exists(local_weights_path)):
    print("Local model weights not found. Defaulting to API Proxy Mode.")
    USE_API_MODE = True

if USE_API_MODE:
    import requests
    HF_TOKEN = os.environ.get("HF_TOKEN")
    print("=" * 60)
    print("RUNNING IN HUGGING FACE API PROXY MODE (Lightweight)")
    print(f"Target Model: {HF_MODEL_ID}")
    if HF_TOKEN:
        print("HF Token: Found (Using authenticated requests)")
    else:
        print("HF Token: NOT found (Using unauthenticated requests - rate limits may apply)")
    print("=" * 60)
else:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("=" * 60)
    print("LOADING KANNADA EMOTION CLASSIFIER LOCALLY (PyTorch)")
    print(f"Path: {MODEL_DIR}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
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
    Run the model (either locally via PyTorch or via HF Inference API)
    and return a dict in the format the frontend expects.
    """
    cleaned = clean_input_text(sentence)

    if USE_API_MODE:
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        try:
            res = requests.post(api_url, headers=headers, json={"inputs": cleaned}, timeout=10)
            if res.status_code != 200:
                # If loading, HF returns estimated_time
                try:
                    err_data = res.json()
                    if isinstance(err_data, dict) and "estimated_time" in err_data:
                        return {
                            "success": False,
                            "error": f"Model is currently loading on Hugging Face. Please try again in {int(err_data['estimated_time'])} seconds."
                        }
                    return {"success": False, "error": err_data.get("error", f"API Error (Status {res.status_code})")}
                except Exception:
                    return {"success": False, "error": f"Hugging Face API returned error status {res.status_code}"}
            
            data = res.json()
            if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
                return {"success": False, "error": "Invalid response format from Hugging Face Inference API"}
            
            predictions = data[0]
            # Sort predictions descending by score
            predictions.sort(key=lambda x: x["score"], reverse=True)
            
            top_pred = predictions[0]
            top_label = top_pred["label"].lower()
            top_score = float(top_pred["score"])
            
            top3 = []
            for item in predictions[:3]:
                top3.append({
                    "emotion": item["label"].lower(),
                    "confidence": f"{item['score']:.2%}",
                    "score": float(item["score"]),
                })
        except Exception as e:
            return {"success": False, "error": f"Proxy request failed: {str(e)}"}
    else:
        # Local inference
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
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        top_idx = int(probs.argmax())
        top_label = ID2LABEL[top_idx]
        top_score = float(probs[top_idx])

        indices_sorted = probs.argsort()[::-1][:3]
        top3 = []
        for i in indices_sorted:
            lbl = ID2LABEL[int(i)]
            score = float(probs[int(i)])
            top3.append({
                "emotion": lbl,
                "confidence": f"{score:.2%}",
                "score": score,
            })

    low_conf = top_score < CONFIDENCE_THRESHOLD

    response = {
        "success": True,
        "original_sentence": sentence,
        "cleaned_sentence": cleaned,
        "predicted_emotion": top_label,
        "confidence": f"{top_score:.2%}",
        "confidence_raw": round(top_score, 4),
        "top_3_predictions": top3,
        "low_confidence": low_conf,
        "threshold": f"{CONFIDENCE_THRESHOLD:.0%}",
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
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask server at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

