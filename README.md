# LLM Inference Optimization Assignment

This repository contains code, configurations, and results from optimizing Llama 3.1 8B-Instruct for production deployment on H100 GPU. This file contains instructions to run the models and also load test and evaluation methodology for each of our models.

## Repository Structure

```text
.
├── README.md                    # This file - setup and methodology
├── RESULTS.md                   # Detailed findings and recommendations
├── requirements.txt             # Python dependencies
├── configs/                     # Server configuration scripts
│   ├── baseline_llama_cpp.sh
│   ├── vllm.sh
│   ├── fp8_quantization.sh
│   ├── sglang.sh
│   └── sglang_torch_compile.sh
├── scripts/                     # Experiment scripts
│   ├── create_dataset.py        # Dataset generation
│   ├── load_test.py             # Load testing script
│   └── evaluate.py              # Quality evaluation script
├── data/                        # Datasets
│   └── test_dataset.json        # Evaluation dataset (520 prompts)
└── results/                     # Experiment results
    ├── results.csv              # Performance comparison table
    ├── load_test_results.csv    # Load test results
    └── evaluation_summary.csv   # Quality evaluation results
```

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (tested on H100 80GB)
- CUDA 12.2+ (required for modern features)

### Software
- Python 3.9+
- Conda (recommended) or virtual environment
- CUDA toolkit 12.2+
- CMake 3.20+ (for llama.cpp)

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n llm_opt python=3.9
conda activate llm_opt

# Install CUDA toolkit (if not already installed)
conda install -c nvidia cuda-toolkit=12.2

# Install all Python dependencies including vLLM and SGLang
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file includes all necessary dependencies including vLLM (v0.11.0) and SGLang (v0.5.4.post3), so no separate framework installation is needed.

### 2. llama.cpp Installation (Baseline)
```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support
conda activate llm_opt
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_COMPILER=$(which nvcc)

cmake --build build --config Release -j --target llama-server

# Convert model to GGUF format
python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B-Instruct \
    --outfile llama-3.1-8b-f16.gguf \
    --outtype f16 \
    --remote
```

## Running Experiments

### 1. Create Test Dataset

```bash
python scripts/create_dataset.py
```

This generates `data/test_dataset.json` with 520 diverse prompts:
- 200 QA prompts from SQuAD v2
- 120 reasoning prompts from GSM8K
- 120 chat prompts from Anthropic's hh-rlhf
- 80 code prompts from HumanEval

### 2. Start Inference Server

Choose one of the configurations from `configs/`:

#### Baseline (llama.cpp)
```bash
chmod +x configs/baseline_llama_cpp.sh
./configs/baseline_llama_cpp.sh
```

#### vLLM
```bash
chmod +x configs/vllm.sh
./configs/vllm.sh
```

#### vLLM FP8 Quantization
```bash
chmod +x configs/fp8_quantization.sh
./configs/fp8_quantization.sh
```

#### SGLang
```bash
chmod +x configs/sglang.sh
./configs/sglang.sh
```

#### SGLang with torch.compile
```bash
chmod +x configs/sglang_torch_compile.sh
./configs/sglang_torch_compile.sh
```

All servers expose OpenAI-compatible API at `http://localhost:8000`

### 3. Run Load Tests

Load tests measure performance metrics under concurrent load:

```bash
# Baseline
python scripts/load_test.py --model baseline_llama_cpp --num-requests 100 --concurrency 8 --csv results/load_test_results.csv

# vLLM
python scripts/load_test.py --model vllm --num-requests 100 --concurrency 8 --csv results/load_test_results.csv

# vLLM FP8
python scripts/load_test.py --model vllm_fp8_quantization --num-requests 100 --concurrency 8 --csv results/load_test_results.csv

# SGLang
python scripts/load_test.py --model sglang --num-requests 100 --concurrency 8 --csv results/load_test_results.csv

# SGLang torch.compile
python scripts/load_test.py --model sglang_torch_compile --num-requests 100 --concurrency 8 --csv results/load_test_results.csv
```

### 4. Run Quality Evaluation

Quality evaluation tests output correctness across task categories:

```bash
# Update scripts/evaluate.py endpoint_url and model_identifier, then run:
python scripts/evaluate.py
```

The script generates outputs for all prompts in `data/test_dataset.json` and evaluates them using task-specific metrics. Results are saved to `results/evaluation_summary.csv`.

## Load Test Methodology

### Overview

The load test (`scripts/load_test.py`) simulates production traffic by sending concurrent requests to the inference server and measuring latency and throughput metrics.

### Implementation Details

**Architecture**:
- Uses `aiohttp` for asynchronous HTTP requests
- Implements semaphore-based concurrency control
- Streams responses to measure Time-to-First-Token (TTFT)

**Request Pattern**:
```python
payload = {
    "prompt": generated_prompt,      # ~2048 tokens
    "max_tokens": 1024,              # Output length
    "temperature": 0.0,              # Deterministic
    "stream": True                   # Enable streaming
}
```

**Metrics**:

1. **TTFT (Time-to-First-Token)**:

2. **Tokens per Second**:
   - Calculated as: `tokens_received / (e2e_latency - ttft)`

3. **End-to-End Latency**:
   - Total time from request submission to response completion
   - Includes all overhead: preprocessing, generation, network

**Test Parameters**:
- **100 requests**: Sufficient sample size for statistical significance
- **Concurrency 8**: Simulates moderate production load
- **2048 input tokens**: Representative of typical prompt lengths
- **1024 output tokens**: Typical response length

**Aggregation**:
- Reports P50 (median) values to capture typical performance
- Filters failed requests from metrics
- Handles timeouts and errors gracefully

## Evaluation Metrics Methodology

### Overview

The evaluation system (`scripts/evaluate.py`) assesses output quality using task-specific metrics designed to measure correctness, coherence, and functionality across different use cases.

### Dataset Composition

The test dataset (`data/test_dataset.json`) contains 520 prompts across 4 categories:

1. **QA (200 prompts)**: Factual question answering from SQuAD v2
2. **Reasoning (120 prompts)**: Mathematical reasoning from GSM8K
3. **Chat (120 prompts)**: Conversational responses from Anthropic's hh-rlhf
4. **Code (80 prompts)**: Code generation from HumanEval

### Few-Shot Learning

The evaluator uses few-shot prompting:
- Extracts first 2 examples from each category
- Builds prompts with category-specific formatting
- Provides context for better model performance

### Task-Specific Metrics

#### 1. QA Evaluation

**Metrics**:
- **Exact Match (EM)**: Binary score based on normalized string comparison
  - Normalization: lowercase, remove punctuation, strip whitespace
  - Returns 1.0 if exact match, 0.0 otherwise
  
- **F1 Score**: Token-level precision/recall harmonic mean
  - Precision: `|predicted_tokens ∩ expected_tokens| / |predicted_tokens|`
  - Recall: `|predicted_tokens ∩ expected_tokens| / |expected_tokens|`
  - F1: `2 * (precision * recall) / (precision + recall)`

**Final Score**: Average of EM and F1 (range: 0.0 to 1.0)

**Rationale**: EM captures exact correctness, F1 captures partial matches useful for factual QA.

#### 2. Reasoning Evaluation

**Metric**: Exact Match on Final Answer

**Process**:
1. Extracts numerical answer from format `#### [answer]`
2. Fallback patterns: "answer is [number]", last number in text
3. Compares extracted numbers with tolerance for floating-point precision
4. Returns 1.0 if match, 0.0 otherwise

**Rationale**: For mathematical reasoning, the final answer is most important. The step-by-step reasoning process can vary while maintaining correctness.

#### 3. Chat Evaluation

**Metric**: Cosine Similarity on Sentence Embeddings

**Process**:
1. Encodes both predicted and expected outputs using `all-MiniLM-L6-v2` embeddings
2. Computes cosine similarity: `cos(θ) = (A · B) / (||A|| * ||B||)`
3. Returns similarity score (range: 0.0 to 1.0)

**Rationale**: Chat responses can vary in wording while maintaining semantic equivalence. Embedding similarity captures semantic coherence and relevance better than exact match.

#### 4. Code Evaluation

**Metric**: Pass@1 (Code Execution Success)

**Process**:
1. Extracts test cases from expected output or prompt
2. Combines prompt (function signature) + generated code + test cases
3. Executes in sandboxed environment using RestrictedPython
4. Returns 1.0 if execution succeeds (no exceptions), 0.0 otherwise

**Rationale**: For code generation, functional correctness is paramount. Pass@1 directly measures whether generated code works correctly.

### Quality Aggregation

**Category Scores**: Average task score across all prompts in category

**Overall Score**: Average of all category scores

**Quality Threshold**: Configurations maintaining overall score within 5% of baseline are considered acceptable.


## Results

See `RESULTS.md` for detailed analysis and `results/results.csv` for performance comparison table.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--gpu-memory-utilization` or `--max-num-seqs`
2. **Port already in use**: Change port in config scripts or stop existing server
3. **Model download fails**: Ensure HuggingFace token is set: `export HF_TOKEN=your_token`
4. **llama.cpp build fails**: Ensure CUDA 12.2+ is available and CMake finds correct compiler



