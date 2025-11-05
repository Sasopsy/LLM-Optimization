#!/bin/bash
# Baseline llama.cpp configuration
# Requires llama.cpp to be built with CUDA support

./build/bin/llama-server \
    --model llama-3.1-8b-f16.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 32768 \
    --parallel 8 \
    --batch-size 4096 \
    --ubatch-size 512 \
    --n-gpu-layers 33 \
    --cont-batching \
    --chat-template llama3 \
    --threads 8

