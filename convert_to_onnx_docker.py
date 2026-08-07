import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    model_id = "premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa"
    onnx_path = "model.onnx"
    quant_path = "model_quant.onnx"
    
    print("=" * 60)
    print("DOCKER BUILD: EXPORTING MODEL TO ONNX & QUANTIZING")
    print("=" * 60)
    
    print(f"Downloading model '{model_id}' from Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    
    print("Exporting model to ONNX format...")
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
    print("Model successfully exported to ONNX.")
    
    print("Quantizing ONNX model to INT8...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"Model successfully quantized. Output saved to {quant_path}")
    
    # Clean up intermediate unquantized model
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

if __name__ == "__main__":
    main()
