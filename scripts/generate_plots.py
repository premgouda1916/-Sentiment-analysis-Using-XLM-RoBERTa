import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import re
import os
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Config ---
LOG_FILE = "xlm-roberta-base model output.txt"
DATA_FILES = {
    "joy": "joy_sentences_2000_kannada.txt",
    "anger": "anger_sentences_2000_kannada.txt",
    "sadness": "sad_sentences_2000_kannada.txt",
    "fear": "fear_sentences_2000_kannada.txt",
    "neutral": "neutral_sentences_2000_kannada.txt"
}
LABELS = ["joy", "anger", "sadness", "fear", "neutral"]

# --- Fig 4.1 Dataset Distribution ---
def plot_dataset_distribution():
    print("Generating Fig 4.1 Dataset Distribution...")
    counts = []
    for label in LABELS:
        filename = DATA_FILES.get(label)
        if filename and os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                # Basic counting: non-empty lines
                count = sum(1 for line in f if line.strip())
                counts.append(count)
        else:
            print(f"Warning: File {filename} not found.")
            counts.append(0)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(LABELS, counts, color=['#F59E0B', '#EF4444', '#3B82F6', '#FB923C', '#4B5563'])
    plt.title('Fig 4.1 Dataset Distribution', fontsize=14)
    plt.xlabel('Emotion Class')
    plt.ylabel('Number of Samples')
    plt.bar_label(bars)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_4.1_Dataset_Distribution.png"))
    plt.close()

# --- Parsing Log File ---
def parse_logs():
    losses = []
    matrix_lines = []
    capture_matrix = False
    
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file {LOG_FILE} not found.")
        return [], []

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse metrics
            # Epoch 1 | Batch 0/625 | Loss: 1.6496
            loss_match = re.search(r"Epoch \d+ \| Batch \d+/\d+ \| Loss: ([\d\.]+)", line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            
            # Parse Confusion Matrix
            if "Confusion matrix (rows=true, cols=pred):" in line:
                capture_matrix = True
                continue
            
            if capture_matrix:
                if line.strip().startswith("[") or line.strip().endswith("]"):
                    matrix_lines.append(line.strip())
                if line.strip().endswith("]]"):
                    capture_matrix = False

    # Process Matrix
    full_matrix_str = "".join(matrix_lines).replace("[", "").replace("]", "")
    # Assuming standard format, we can split by whitespace
    matrix_values = [int(x) for x in full_matrix_str.split()]
    # 5 classes -> 5x5 matrix
    if len(matrix_values) == 25:
        confusion_matrix = np.array(matrix_values).reshape(5, 5)
    else:
        print("Error: Could not parse confusion matrix correctly.")
        confusion_matrix = None
        
    return losses, confusion_matrix

# --- Fig 4.3 Training Loss Curve ---
def plot_training_loss(losses):
    print("Generating Fig 4.3 Training Loss Curve...")
    if not losses:
        print("No loss data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='#4F46E5', alpha=0.9, linewidth=1.5)
    # Smooth curve
    if len(losses) > 20:
        window_size = 20
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        # Adjust x-axis for moving avg
        plt.plot(range(window_size-1, len(losses)), moving_avg, label='Smoothed Loss (MA=20)', color='#DC2626', linewidth=2)
    
    plt.title('Fig 4.3 Training Loss Curve', fontsize=14)
    plt.xlabel('Training Steps (Batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_4.3_Training_Loss.png"))
    plt.close()

# --- Fig 5.1 Confusion Matrix ---
def plot_confusion_matrix(matrix):
    print("Generating Fig 5.1 Confusion Matrix...")
    if matrix is None:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Fig 5.1 Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_5.1_Confusion_Matrix.png"))
    plt.close()

# --- Fig 1.2 Project Workflow (Flowchart) ---
def plot_project_workflow():
    print("Generating Fig 1.2 Project Workflow...")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    arrow_props = dict(facecolor='#1F2937', edgecolor='#1F2937', arrowstyle='->', lw=1.5)

    def draw_box(x, y, text, w=20, h=5, color='#DBEAFE'):
        # x, y is center
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.2', 
                                      linewidth=1, edgecolor='#3B82F6', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='#1E3A8A')
        return x, y, h

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

    # Title
    ax.text(50, 98, "Fig 1.2 Project Workflow", ha='center', fontsize=16, fontweight='bold')

    # --- Data Preparation Section ---
    rect_data = patches.FancyBboxPatch((5, 75), 90, 20, boxstyle='round,pad=0.5', ec='#93C5FD', fc='#EFF6FF')
    ax.add_patch(rect_data)
    ax.text(10, 93, "Data Preparation", fontsize=11, fontweight='bold', color='#1D4ED8')

    draw_box(30, 85, "Raw Dataset", w=16)
    draw_box(50, 85, "Preprocessing", w=16)
    draw_box(70, 85, "Tokenization", w=16)
    draw_arrow(38, 85, 42, 85)
    draw_arrow(58, 85, 62, 85)

    draw_box(70, 78, "Train/Val/Test Split", w=20, color='#FEF3C7')
    draw_arrow(70, 82.5, 70, 80.5)

    # --- Training Section ---
    rect_train = patches.FancyBboxPatch((5, 45), 90, 28, boxstyle='round,pad=0.5', ec='#FCA5A5', fc='#FEF2F2')
    ax.add_patch(rect_train)
    ax.text(10, 71, "Training Phase", fontsize=11, fontweight='bold', color='#B91C1C')

    draw_box(30, 65, "Load Model", w=16)
    draw_box(50, 65, "Fine-tune (3 Epochs)", w=20, color='#FCA5A5')
    draw_arrow(38, 65, 42, 65)
    # Split to Finetune
    ax.annotate('', xy=(50, 67.5), xytext=(70, 75.5), arrowprops=arrow_props)

    draw_box(50, 55, "Evaluation", w=15, color='#FCD34D')
    draw_arrow(50, 62.5, 50, 57.5)

    draw_box(50, 48, "Save Model", w=16, color='#D1FAE5')
    draw_arrow(50, 52.5, 50, 50.5)

    # --- Web App Section ---
    rect_app = patches.FancyBboxPatch((5, 5), 90, 38, boxstyle='round,pad=0.5', ec='#6EE7B7', fc='#ECFDF5')
    ax.add_patch(rect_app)
    ax.text(10, 41, "Web Application", fontsize=11, fontweight='bold', color='#047857')

    draw_box(20, 35, "User Input", w=15)
    draw_box(40, 35, "Preprocessing", w=15)
    draw_box(60, 35, "Tokenization", w=15)
    draw_arrow(27.5, 35, 32.5, 35)
    draw_arrow(47.5, 35, 52.5, 35)

    draw_box(60, 25, "Inference", w=15, color='#FCA5A5')
    draw_arrow(60, 32.5, 60, 27.5)
    
    # Model to Inference (Dashed)
    ax.annotate('', xy=(58, 27.5), xytext=(50, 45), 
                arrowprops=dict(arrowstyle='->', linestyle='dashed', color='#6B7280'))
    
    draw_box(60, 15, "Softmax", w=15)
    draw_arrow(60, 22.5, 60, 17.5)

    draw_box(60, 8, "Final Prediction", w=18, color='#C4B5FD')
    draw_arrow(60, 12.5, 60, 10.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_1.2_Project_Workflow.png"), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_project_workflow()
    plot_dataset_distribution()
    losses, cm = parse_logs()
    plot_training_loss(losses)
    plot_confusion_matrix(cm)
    print(f"Done. Figures saved to {OUTPUT_DIR}/")
