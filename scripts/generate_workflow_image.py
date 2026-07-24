import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_flowchart():
    fig, ax = plt.subplots(figsize=(14, 20)) # Increased figure size to accommodate larger text
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Font sizes
    FONT_SIZE_BOX = 18
    FONT_SIZE_TITLE = 20

    # Function to draw box
    def draw_box(x, y, width, height, text, color='lightblue'):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                      linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=FONT_SIZE_BOX, wrap=True)
        return x + width/2, y, y + height # Return connection points (center_x, bottom_y, top_y)

    # Function to draw arrow
    def draw_arrow(x1, y1, x2, y2):
        # Ensure straight lines where possible
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # Phase 1: Data Preparation
    ax.text(0.5, 19.5, "Phase 1: Data Preparation", fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='left')
    rect1 = patches.Rectangle((0.2, 13.2), 9.6, 6.5, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect1)

    # Adjusted positions for alignment (Center X = 5)
    cx, b_raw, t_raw = draw_box(2.5, 18.2, 5, 1.2, "Raw Dataset") # Centered, wider
    cx, b_pre, t_pre = draw_box(2.5, 16.4, 5, 1.2, "Preprocessing")
    cx, b_tok, t_tok = draw_box(2.5, 14.6, 5, 1.2, "XLM-R Tokenizer")
    
    # Split
    cx_split, b_split, t_split = draw_box(2.5, 13.5, 5, 0.8, "Dataset Split", color='lightyellow')
    
    draw_arrow(cx, b_raw, cx, t_pre) # Vertical down
    draw_arrow(cx, b_pre, cx, t_tok) # Vertical down
    draw_arrow(cx, b_tok, cx, t_split) # Vertical down
    
    # Split outputs
    cx_train, b_train, t_train = draw_box(0.5, 11.5, 3, 1.2, "Training Set\n(80%)", color='lightgreen')
    cx_val, b_val, t_val = draw_box(3.8, 11.5, 2.4, 1.2, "Val Set\n(10%)", color='lightgreen') # Middle
    cx_test, b_test, t_test = draw_box(6.5, 11.5, 3, 1.2, "Testing Set\n(10%)", color='lightgreen') 
    
    # Arrows from split - adjusting to come from bottom center of split box
    draw_arrow(cx_split, b_split, cx_train, t_train)
    draw_arrow(cx_split, b_split, cx_val, t_val)
    draw_arrow(cx_split, b_split, cx_test, t_test)

    # Phase 2: Training
    ax.text(0.5, 10.8, "Phase 2: Training Phase", fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='left')
    rect2 = patches.Rectangle((0.2, 7.2), 6.1, 3.8, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect2)
    
    cx_base, b_base, t_base = draw_box(0.5, 9.5, 2.5, 1.2, "Pre-trained\nXLM-R")
    cx_ft, b_ft, t_ft = draw_box(0.5, 7.5, 2.5, 1.2, "Fine-tuning\nLoop", color='salmon')
    
    cx_eval, b_eval, t_eval = draw_box(3.8, 8.5, 2.4, 1.2, "Evaluation", color='salmon')
    cx_save, b_save, t_save = draw_box(3.8, 7.3, 2.4, 0.8, "Save Model", color='gold')

    draw_arrow(cx_base, b_base, cx_ft, t_ft)
    draw_arrow(cx_train, b_train, cx_ft, t_ft) 
    
    # Loop arrow logic for fine-tuning to eval
    # Making it simpler: FT -> Eval
    draw_arrow(cx_ft, 8.1, cx_eval, 8.1) # Horizontal
    draw_arrow(cx_val, b_val, cx_eval, t_eval) 
    draw_arrow(cx_eval, b_eval, cx_save, t_save)

    # Phase 3: Testing
    ax.text(6.8, 10.8, "Phase 3: Testing", fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='left')
    rect3 = patches.Rectangle((6.4, 7.2), 3.4, 3.8, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect3)
    
    cx_testinf, b_testinf, t_testinf = draw_box(6.6, 9.2, 3, 1.2, "Inference\n(Unseen)")
    cx_met, b_met, t_met = draw_box(6.6, 7.5, 3, 1.2, "Metrics\nReport")
    
    draw_arrow(cx_test, b_test, cx_testinf, t_testinf) 
    draw_arrow(cx_save, 7.7, cx_testinf, 9.2) # Diagonal connection
    draw_arrow(cx_testinf, b_testinf, cx_met, t_met)


    # Phase 4: Web Application
    ax.text(0.5, 6.5, "Phase 4: Web Application (Inference)", fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='left')
    rect4 = patches.Rectangle((0.2, 0.2), 9.6, 6.5, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect4)

    # Center these boxes at X=5
    cx_user, b_user, t_user = draw_box(2.5, 5.0, 5, 1.2, "User Input (Text)")
    cx_wpre, b_wpre, t_wpre = draw_box(2.5, 3.5, 5, 1.2, "Preprocessing")
    cx_wtok, b_wtok, t_wtok = draw_box(2.5, 2.0, 5, 1.2, "Tokenization")
    cx_winf, b_winf, t_winf = draw_box(0.5, 0.5, 4, 1.2, "Inference Engine", color='lightblue')
    cx_out, b_out, t_out = draw_box(5.5, 0.5, 4, 1.2, "Final Prediction", color='lightgreen')
    
    draw_arrow(cx_user, b_user, cx_wpre, t_wpre)
    draw_arrow(cx_wpre, b_wpre, cx_wtok, t_wtok)
    draw_arrow(cx_wtok, b_wtok, cx_winf, t_winf) # Diagonal
    draw_arrow(cx_winf, 0.5+0.6, cx_out, 0.5+0.6) # Horizontal from engine to prediction

    plt.tight_layout()
    plt.savefig('figures/Fig_1.2_Project_Workflow_Updated.png', dpi=300, bbox_inches='tight')
    print("Flowchart generated successfully.")

if __name__ == "__main__":
    create_flowchart()
