# Kannada Emotion Classification using XLM-RoBERTa

A state-of-the-art Natural Language Processing (NLP) application to classify emotions in Kannada text sentences using a fine-tuned **XLM-RoBERTa** model. The project features a web-based user interface built with Flask and Vanilla CSS for real-time predictions and detailed analytics.

---

## 🚀 Key Features

*   **Fine-tuned XLM-RoBERTa Model**: Specifically optimized for Kannada syntax, vocabulary, and emotional expressions.
*   **5-Class Emotion Classification**: Detects **Joy (ಸಂತೋಷ)**, **Anger (ಕೋಪ)**, **Sadness (ದುಃಖ)**, **Fear (ಭಯ)**, and **Neutral (ಸಾಮಾನ್ಯ)**.
*   **Web Dashboard**: A beautiful, user-friendly interface to test predictions in real-time.
*   **Top-3 Predictions**: View confidence scores for secondary emotions.
*   **Data Augmentation Script**: Includes helper scripts for adding keywords and boosting performance on low-resource training scenarios.

---

## 📁 Repository Structure

```text
├── documents/                       # Project documentation & presentations
│   ├── A3_Poster_Example.pptx
│   ├── Mini_Project_Report_Kannada_Emotion_Classification.docx
│   └── Mini_Project_team_8_final_ppt.pptx
│
├── kannada_emotion_model_xlm/       # Saved Model & Configuration (exclude weights in git)
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── templates/                       # Flask Frontend templates
│   └── index.html                   # Web interface
│
├── figures/                         # Evaluation figures & workflow plots
│   ├── Fig_1.2_Project_Workflow.png
│   └── Fig_5.1_Confusion_Matrix.png
│
├── scripts/                         # Visualization scripts
│   ├── generate_plots.py
│   └── generate_workflow_image.py
│
├── app.py                           # Flask backend web application
├── training.py                      # Model fine-tuning & training script
├── test_model_direct.py             # CLI script for quick model inference testing
├── augment_with_keywords.py         # Keyword-based data augmentation helper
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignored folders and files
└── README.md                        # Documentation (this file)
```

---

## 🛠️ Getting Started

### Prerequisites

*   Python 3.8 or higher
*   PIP (Python Package Installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/kannada-emotion-xlmr.git
    cd kannada-emotion-xlmr
    ```

2.  **Create and activate a Virtual Environment**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Running the Web Application

1.  **Run the Flask Server**:
    ```bash
    python app.py
    ```
2.  **Access the Dashboard**:
    Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🏋️ Training & Augmentation

*   **Data Augmentation**: To append emotion-specific keywords and template variants to boost model accuracy, configure the folder path in `augment_with_keywords.py` and run:
    ```bash
    python augment_with_keywords.py
    ```
*   **Model Training**: Ensure the five emotion dataset text files are in the root directory, then start training/fine-tuning:
    ```bash
    python training.py
    ```

---

## 📊 Model Details & Architecture

The classification head is fine-tuned on top of the multilingual **XLM-RoBERTa (Base)** architecture:
*   **Base Model**: `xlm-roberta-base` (12-layer Transformer encoder)
*   **Max Sequence Length**: 64 tokens
*   **Output Node**: 5 classes with Softmax activation
*   **Optimizers**: AdamW with weight decay (`0.01`), learning rate `2e-5`
