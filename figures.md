# Project Figures

## Fig 1.1 System Architecture
```mermaid
graph TD
    A[User Input (Text)] --> B[Preprocessing]
    B --> C{Cleanup}
    C --> D[Tokenization (XLM-R Tokenizer)]
    D --> E[XLM-RoBERTa Model]
    E --> F[Classification Head]
    F --> G[Emotion Prediction]
    G --> H[Web Interface Display]
```

## Fig 1.2 Project Workflow
![Project Workflow](figures/Fig_1.2_Project_Workflow_Updated.png)

### Flowchart Explanation

The project workflow is logically structured into four distinct phases, guiding the process from raw data to a deployable application: **Data Preparation**, **Training**, **Testing**, and the **Web Application**.

**1. Data Preparation Phase**
This foundational phase transforms raw text into a format suitable for the model.
*   **Raw Dataset**: The process begins with a corpus of Kannada sentences labeled with five emotions (Joy, Anger, Sadness, Fear, Neutral).
*   **Preprocessing**: The raw text is cleaned to remove noise, including zero-width joiners (ZWJ), special characters, and extra whitespace, ensuring high-quality input.
*   **Tokenization**: The cleaned text is passed through the **XLM-RoBERTa Tokenizer**. This step converts sentences into sub-word tokens and maps them to numerical IDs (Input IDs) while generating Attention Masks to handle variable sequence lengths.
*   **Dataset Split**: The tokenized data is stratified and split into three sets: **Training (80%)** for model learning, **Validation (10%)** for hyperparameter tuning, and **Testing (10%)** for final evaluation.

**2. Training Phase**
In this phase, the model learns to recognize emotions.
*   **Model Initialization**: A pre-trained **XLM-RoBERTa Base** model is loaded. This model already understands multilingual linguistic patterns.
*   **Fine-tuning**: The model is fine-tuned on the Training Set for 3 epochs. During this loop, the model updates its weights to minimize the loss on the specific emotion classification task.
*   **Evaluation**: After each epoch, the model is evaluated on the Validation Set using metrics like Accuracy and F1-Score to monitor progress and prevent overfitting.
*   **Saving**: The best-performing model and its corresponding tokenizer are serialized and **saved** to the local disk, creating the `saved_model` directory.

**3. Testing Phase**
This phase provides an unbiased assessment of the final model.
*   **Model Inference**: The saved, fine-tuned model is loaded and applied to the held-out **Testing Set**. Crucially, the model has never seen this data during training.
*   **Performance Metrics**: The model's predictions are compared against ground-truth labels to generate a **Confusion Matrix** and a detailed classification report. This step confirms the model's generalization capability before real-world theoretical deployment.

**4. Web Application Layer**
The final phase is the user-facing deployment.
*   **User Input**: A user enters a Kannada sentence into the web interface.
*   **Real-time Processing**: The input undergoes the same **Preprocessing** and **Tokenization** pipeline used during training to ensure consistency.
*   **Inference Engine**: The processed inputs are fed into the loaded **XLM-RoBERTa model** to obtain raw logic outputs (logits).
*   **Prediction**: A **Softmax** function converts logits into probability scores for each emotion class. The system displays the emotion with the highest probability as the **Final Prediction** to the user.

## Fig 4.2 XLM-RoBERTa Architecture
```mermaid
graph TD
    subgraph "XLM-RoBERTa Base"
    Input[Input IDs & Mask] --> Embed[Embeddings Layer]
    Embed --> Enc1[Transformer Encoder 1]
    Enc1 --> Enc2[Transformer Encoder 2]
    Enc2 --> Enc3[...]
    Enc3 --> Enc12[Transformer Encoder 12]
    end
    Enc12 --> Pool[Pooling Layer]
    Pool --> Drop[Dropout 0.1]
    Drop --> Dense[Dense Layer]
    Dense --> Act[Tanh Activation]
    Act --> OutDropout[Dropout 0.1]
    OutDropout --> Class[Classifier (5 Output Nodes)]
    Class -- Softmax --> Prob[Probabilities]
```
