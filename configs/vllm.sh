#!/bin/bash
# vLLM configuration (optimization 1: serving framework)

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9

