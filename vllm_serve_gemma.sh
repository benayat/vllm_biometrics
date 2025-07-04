#!/bin/bash

# vLLM OpenAI Compatible Server Script
# This script serves the multimodal model as an OpenAI-compatible API server

MODEL_NAME="RedHatAI/gemma-3-27b-it-quantized.w4a16"
HOST="0.0.0.0"
PORT="8000"
MAX_MODEL_LEN="2048"

echo "Starting vLLM OpenAI-compatible server..."
echo "Model: $MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"

# Start the vLLM OpenAI API server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --served-model-name "gemma-3-27b-it" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "gemma-3-27b-it" \
    --max-parallel-loading-workers 8 \
    --disable-log-requests

echo "Server started at http://$HOST:$PORT"
echo "API documentation available at http://$HOST:$PORT/docs"