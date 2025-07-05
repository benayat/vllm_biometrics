#!/bin/bash

# Optimized vLLM OpenAI-compatible Server for Maximum Concurrency & Throughput
MODEL_NAME="RedHatAI/gemma-3-27b-it-quantized.w4a16"
HOST="0.0.0.0"
PORT="8000"
MAX_MODEL_LEN="2048"

# Increased batch settings for maximum utilization
MAX_BATCHED_TOKENS="65536"     # Increase to batch more tokens concurrently
MAX_NUM_SEQS="2048"            # Higher concurrency limit

echo "Starting optimized vLLM OpenAI-compatible server..."
echo "Model: $MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Max Num Sequences: $MAX_NUM_SEQS"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --served-model-name "gemma-3-27b-it" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-log-requests

echo "Optimized server running at http://$HOST:$PORT"
echo "API documentation available at http://$HOST:$PORT/docs"