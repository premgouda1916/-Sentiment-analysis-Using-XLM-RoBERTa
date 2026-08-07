from flask import Flask, request, jsonify, render_template
import os
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "model_quant.onnx")

# We load tokenizer from the Hugging Face repo ID directly, which only downloads
# a few small config JSON files (~10KB) during startup and caches them.
HF_MODEL_ID = "premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa"

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
# LOAD ONNX MODEL & TOKENIZER
# --------------------------------------------------

print("=" * 60)
print("LOADING QUANTIZED KANNADA EMOTION CLASSIFIER (ONNX Runtime)")
print("=" * 60)

if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"Quantized model not found at: {ONNX_MODEL_PATH}")

# Load the ONNX model session
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print(f"[OK] ONNX Session created successfully.")

# Load the tokenizer
print("Loading tokenizer from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
print("[OK] Tokenizer loaded successfully.")
print("=" * 60)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_input_text(text: str) -> str:
    """Light cleaning: normalize spaces, remove ZWJ etc."""
    text = text.replace("\u200c", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)


def classify_sentence(sentence: str):
    """
    Run the ONNX model and return a dict in the same format
    your frontend expects.
    """
    cleaned = clean_input_text(sentence)

    # Tokenize input (using numpy arrays for ONNX Runtime)
    encoded = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="np"
    )

    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

    try:
        # Run inference in ONNX Runtime
        outputs = ort_session.run(None, inputs)
        logits = outputs[0][0]  # shape: [num_labels]
        probs = softmax(logits)
        
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
    except Exception as e:
        print("❌ Error during ONNX classification:", e)
        return {
            "success": False,
            "error": f"Inference execution failed: {str(e)}"
        }


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
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
        print("❌ Error during prediction API:", e)
        return jsonify({
            "success": False,
            "error": "Server error during prediction."
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_format": "onnx_quantized",
        "model_path": ONNX_MODEL_PATH
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask server at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
