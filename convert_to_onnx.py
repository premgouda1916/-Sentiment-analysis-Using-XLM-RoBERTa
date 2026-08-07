import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    print("=" * 60)
    print("CONVERTING MODEL TO ONNX & QUANTIZING TO INT8")
    print("=" * 60)
    
    model_dir = "kannada_emotion_model_xlm"
    onnx_path = "model.onnx"
    quant_path = "model_quant.onnx"
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] Local model directory '{model_dir}' not found!")
        sys.exit(1)
        
    print("Loading PyTorch model and tokenizer...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
        
    print("Exporting model to ONNX format...")
    try:
        # Create dummy input for tracing
        dummy_text = "ನನಗೆ ತುಂಬಾ ಸಂತೋಷವಾಗಿದೆ"
        inputs = tokenizer(dummy_text, return_tensors="pt")
        
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        print(f"[OK] Exported successfully to {onnx_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export to ONNX: {e}")
        sys.exit(1)
        
    print("\nQuantizing ONNX model to INT8...")
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"[OK] Quantized successfully to {quant_path}")
        
        # Print file size comparison
        orig_size = os.path.getsize(os.path.join(model_dir, "model.safetensors")) / (1024 * 1024)
        quant_size = os.path.getsize(quant_path) / (1024 * 1024)
        print(f"\nModel Size Comparison:")
        print(f"  - Original PyTorch Model: {orig_size:.2f} MB")
        print(f"  - Quantized ONNX Model  : {quant_size:.2f} MB (saved {orig_size - quant_size:.2f} MB!)")
        
        # Clean up unquantized model to save space
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print("Cleaned up unquantized model.onnx")
            
    except ImportError:
        print("[ERROR] onnx or onnxruntime package not installed. Run 'pip install onnx onnxruntime'")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
