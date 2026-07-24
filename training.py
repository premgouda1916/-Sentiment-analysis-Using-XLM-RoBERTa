import os
import re
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- CONFIG ----------------

MODEL_NAME = "xlm-roberta-base"

# Folder where the trained model will be saved
MODEL_SAVE_DIR = "./kannada_emotion_model_xlm"

# Your 2000-sentence files
DATA_FILES: Dict[str, str] = {
    "joy":     "joy_sentences_2000_kannada.txt",
    "anger":   "anger_sentences_2000_kannada.txt",
    "sadness": "sad_sentences_2000_kannada.txt",
    "fear":    "fear_sentences_2000_kannada.txt",
    "neutral": "neutral_sentences_2000_kannada.txt",
}

# Fixed numeric mapping
EMOTION_LABELS: Dict[str, int] = {
    "joy": 0,
    "anger": 1,
    "sadness": 2,
    "fear": 3,
    "neutral": 4,
}
ID2LABEL = {v: k for k, v in EMOTION_LABELS.items()}
NUM_CLASSES = len(EMOTION_LABELS)

MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# ----------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    # minimal cleaning – keep Kannada as-is
    text = text.replace("\u200c", "")          # remove ZWJ if present
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(files_dict: Dict[str, str], labels_dict: Dict[str, int]):
    texts, labels = [], []
    for emotion, path in files_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        cleaned = [clean_text(l) for l in lines]
        # Remove empty strings
        cleaned = [c for c in cleaned if c]

        if len(cleaned) == 0:
            raise ValueError(f"No valid sentences for '{emotion}' in {path}")

        texts.extend(cleaned)
        labels.extend([labels_dict[emotion]] * len(cleaned))

        print(f"Loaded {len(cleaned):4d} sentences for '{emotion}' from {path}")

    return np.array(texts), np.array(labels)


class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, data_loader, device):
    model.eval()
    losses = []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1_macro, all_targets, all_preds


def main():
    print("=" * 70)
    print("KANNADA EMOTION CLASSIFIER – DistilBERT (2000 per class)")
    print("=" * 70)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("-" * 70)

    set_seed(SEED)

    # ---------- LOAD DATA ----------
    texts, labels = load_data(DATA_FILES, EMOTION_LABELS)

    print(f"\nTotal samples: {len(texts)}")
    print("\nClass distribution:")
    for emo, lid in EMOTION_LABELS.items():
        cnt = int((labels == lid).sum())
        print(f"  {emo:7s}: {cnt:4d}")

    # 80/10/10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels,
        test_size=0.10,
        random_state=SEED,
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.1111,           # ~10% of total
        random_state=SEED,
        stratify=y_temp,
    )

    print("\nSplit sizes:")
    print("  Train:", len(X_train))
    print("  Val  :", len(X_val))
    print("  Test :", len(X_test))

    # ---------- DATASETS ----------
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmotionDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LENGTH)
    val_ds   = EmotionDataset(X_val.tolist(),   y_val.tolist(),   tokenizer, MAX_LENGTH)
    test_ds  = EmotionDataset(X_test.tolist(),  y_test.tolist(),  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False)

    # ---------- MODEL ----------
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=EMOTION_LABELS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = max(EPOCHS * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )

    # ---------- TRAIN ----------
    best_val_f1 = 0.0
    best_state = None

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = float(np.mean(train_losses))
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train loss : {avg_train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val acc    : {val_acc:.4f}")
        print(f"  Val F1(m)  : {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("  👉 New BEST model (by Val F1)")
        print("-" * 60)

    # ---------- TEST ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"\nLoaded best model (Val F1={best_val_f1:.4f}) for testing.\n")

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)

    print("TEST RESULTS")
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc:.4f}")
    print(f"  Test F1(m): {test_f1:.4f}")

    print("\nClassification report (TEST):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[k for k,_ in sorted(EMOTION_LABELS.items(), key=lambda x: x[1])],
            digits=4,
        )
    )

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    # ---------- SAVE ----------
    print(f"\nSaving model + tokenizer to: {MODEL_SAVE_DIR}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    with open(os.path.join(MODEL_SAVE_DIR, "id2label.txt"), "w", encoding="utf-8") as f:
        for i in range(NUM_CLASSES):
            f.write(f"{i}\t{ID2LABEL[i]}\n")

    print("\n✅ DONE – model saved in kannada_emotion_model_2000")
    print("=" * 70)


if __name__ == "__main__":
    main()
