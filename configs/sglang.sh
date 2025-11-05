#!/bin/bash
# SGLang configuration (optimization 3: alternative serving framework)

python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-running-requests 256 \
    --max-prefill-tokens 4096 \
    --mem-fraction-static 0.9

