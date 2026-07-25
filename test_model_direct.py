from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MODEL_DIR = "./kannada_emotion_model_xlm"
# Try to load local model first. If model.safetensors is missing (e.g. on GitHub), fall back to Hugging Face Hub.
HF_MODEL_ID = "premgouda1916/kannada_sentiment_classifier_Xlm_RoBERTa" # <-- Replace with your actual Hugging Face ID!

ID2LABEL = {
    0: "joy",
    1: "anger",
    2: "sadness",
    3: "fear",
    4: "neutral",
}

import os
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

print(f"Model loaded on device: {device}")
print(f"Model config id2label: {model.config.id2label}")

# Test sentences
test_sentences = [
    "ನನಗೆ ಈ ನಿರ್ಧಾರದ ತುಂಬಾ ಹೆಮ್ಮೆಯಾಗಿದೆ",  # Should be joy
    "ನನಗೆ ಯೋಜನೆಯಲ್ಲಿ ಹೆಚ್ಚಿನ ತುಂಬಾ ಇಷ್ಟ",  # Should be joy
    "ನನಗೆ ಖುಷಿ", # Joy (Short)
    "ತುಂಬಾ ಕೋಪ", # Anger (Short)
    "ಭಯವಾಗುತ್ತಿದೆ", # Fear (Short)
    "ಬೇಸರ", # Sadness (Short)
    "ಸಂತೋಷ", # Joy (Keyword)
    "ನನಗೆ ಕಬ್ಬಡ್ಡಿ ಎಂದರೆ ಬಹಳ ಇಷ್ಟ", # User query
    "ನನಗೆ ಹಸಿವಾಗುತ್ತಿದೆ", # User error: Hungry -> Joy?
    "ನನಗೆ ಹೊಟ್ಟೆ ಹಸಿವಾಗುತ್ತಿದೆ", # User error: Stomach hungry -> Joy?
    "ಮಂಗಳೂರು ಒಂದು ಸುಂದರ ನಗರ ಆದರೆ ಅದರಲ್ಲಿ ತುಂಬಾ ಕೆಟ್ಟ ಶಕ್ತಿ ಇದೆ", # User error: Mixed -> Joy?
]

for sentence in test_sentences:
    print(f"\nTesting: {sentence}")
    
    encoded = tokenizer(
        sentence,
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
    
    print(f"  Predicted: {top_label} ({top_score:.2%})")
    print(f"  All probabilities:")
    for i, prob in enumerate(probs):
        print(f"    {ID2LABEL[i]}: {prob:.2%}")
