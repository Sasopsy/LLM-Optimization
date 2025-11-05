#!/bin/bash
# vLLM with FP8 quantization (optimization 2: quantization)

python -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --dtype auto \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9

