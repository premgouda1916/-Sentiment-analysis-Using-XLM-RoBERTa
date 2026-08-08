# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.9-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install packages required for conversion
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.39.3 \
    onnx==1.15.0 \
    onnxruntime==1.17.3

# Copy only the conversion script
COPY convert_to_onnx_docker.py .

# Run conversion script to download the model from HF and export/quantize it to model_quant.onnx
RUN python convert_to_onnx_docker.py

# ==========================================
# STAGE 2: RUNNER
# ==========================================
FROM python:3.9-slim AS runner

WORKDIR /app

# Install runtime dependencies (no torch!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the quantized model from builder stage
COPY --from=builder /build/model_quant.onnx .

# Copy application files
COPY app.py .
COPY templates/ templates/

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
