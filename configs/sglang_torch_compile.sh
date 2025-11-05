#!/bin/bash
# SGLang with torch.compile (optimization 4: configuration tuning)

python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-torch-compile \
    --host 0.0.0.0 \
    --port 8000 \
    --max-running-requests 256 \
    --max-prefill-tokens 4096 \
    --mem-fraction-static 0.9

