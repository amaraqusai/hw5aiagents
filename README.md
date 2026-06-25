# HW5 AI Agents: Large LLM Inference on Consumer Hardware

This project demonstrates how to run a massive Large Language Model (LLM) — specifically **Qwen1.5-14B** (which normally requires 28+ GB of RAM) — on standard consumer hardware with only 16 GB of RAM, without crashing due to Out-Of-Memory (OOM) errors.

## 🧠 The Problem
Loading an LLM with 14 Billion parameters using standard HuggingFace (`transformers`) attempts to place the *entire model* into memory (RAM or VRAM) simultaneously. 
For a 14B model in `fp16` precision, this requires approximately 28 GB of memory. On a machine with only 16 GB of RAM, the operating system attempts to use disk swap space, leading to an immediate **OOM Crash**.

## 🛠️ The Solution: AirLLM Disk Streaming
This project utilizes [AirLLM](https://github.com/lyogavin/Anima/tree/main/air_llm), a library that drastically reduces memory requirements for LLM inference.

**How does AirLLM work?**
Instead of loading the full model, AirLLM breaks the model down and uses a **Layer-by-Layer Disk Streaming** technique:
1. **Sharding**: The model is split into 40 individual layers and saved to disk.
2. **Streaming**: During inference, only **one layer** is loaded into RAM at a time.
3. **Execution**: The input passes through the loaded layer, the output is kept, and the layer is discarded from RAM.
4. **Repetition**: The next layer is loaded from disk, and the process repeats.

**The Tradeoff:**
By using the disk as an extension of memory, we drop our peak RAM usage from **28+ GB down to ~2.03 GB**. However, disk I/O is vastly slower than RAM, meaning inference time increases from seconds to minutes.

---

## 🔧 Deep Dive: The Qwen "Monkeypatch" Fix
During implementation, we encountered a critical compatibility issue. AirLLM was designed for `transformers v4.36`, but Qwen models (`Qwen2Attention`) in newer `transformers` versions fundamentally changed how they handle Positional/Rotary Embeddings. 

Standard AirLLM passes a 1D `position_ids` tensor, but new Qwen models expect pre-calculated `position_embeddings` to be passed as a tuple. This mismatch resulted in a `TypeError: Qwen2Attention.forward() missing 1 required positional argument: 'position_embeddings'`.

To fix this *without* downgrading libraries or modifying external packages, we implemented a runtime **Monkeypatch** in `airllm_cpu_inference.py`:

1. **Intercepting the Forward Pass**: We completely replaced the `forward` function of `Qwen2Attention`.
2. **Inline Calculation**: We initialize a `Qwen2RotaryEmbedding` layer dynamically. When AirLLM passes `position_ids`, our patched function calculates the `cos` and `sin` embeddings *on the fly*.
3. **Tensor Slicing**: We intercept the `attention_mask` (which AirLLM sends as `[1, 1, 512, 512]`) and slice it to match the actual sequence length of the incoming query/key states, preventing `RuntimeError` during Scaled Dot-Product Attention (SDPA).
4. **Logits Reshaping**: AirLLM returns 2D logits `[seq_len, vocab_size]`, but the `transformers` generation loop expects 3D `[batch, seq_len, vocab_size]`. We patched `AirLLMBaseModel.forward` to unsqueeze a batch dimension.

---

## 📁 Project Structure
```
hw5aiagents/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── TODO.md                            # Assignment checklist with results
├── airllm_cpu_inference.py            # [MAIN] AirLLM CPU inference with monkeypatches
├── baseline_oom.py                    # Baseline OOM crash demonstration
├── ollama_inference.py                # Ollama local inference benchmark
├── benchmark_utils.py                 # Benchmarking utility class + final results table
├── colab_full_benchmark.py            # Consolidated script for Google Colab
├── HW5_AI_Agents_Full_Benchmark.ipynb # Jupyter Notebook version
└── logs/
    └── baseline_failure.txt           # OOM crash evidence log
```

---

## ⚙️ Prerequisites
- **Python** 3.10 or 3.11 (not 3.12+)
- **16 GB RAM** (minimum)
- **~40 GB free disk space** (for model download + layer sharding)
- **Ollama** installed locally (for Ollama benchmark test)

## 📦 Installation
```bash
# 1. Clone the repository
git clone https://github.com/amaraqusai/hw5aiagents.git
cd hw5aiagents

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Ollama Setup (for Test 3)
```bash
# 1. Download and install Ollama from https://ollama.com
# 2. Verify installation
ollama --version

# 3. Pull the llama2 model
ollama pull llama2

# 4. Run the Ollama server (runs in background)
ollama serve
```

---

## 🚀 How to Run

### 1. The Standard Baseline (Expected to Crash)
This script attempts to load the 14B model standardly. It is designed to prove that the hardware cannot natively support the model.
```bash
python baseline_oom.py
```
*Expected Output: `[X] OOM CRASH`*

### 2. The AirLLM Inference (The Solution)
This script applies the monkeypatches and streams the layers from disk. Note: The first run will take ~30-60 minutes as it downloads and shards the ~28GB model. Subsequent runs are faster.
```bash
python airllm_cpu_inference.py
```
*Expected Output: Successful text generation using < 3GB RAM.*

### 3. The Final Benchmark
Generates a clean comparison table of the results.
```bash
python benchmark_utils.py
```

---

## 📊 Benchmark Results

| Framework | Model | TTFT (s) | Total Runtime (s) | Peak RAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HuggingFace (Standard)** | Qwen1.5-14B | CRASHED | CRASHED | 28+ (OOM) | [X] OOM CRASH |
| **Ollama (Local)** | llama2-7B (Q4) | 18.23 | 136.3 | 0.25 | [OK] SUCCESS |
| **AirLLM (CPU Streaming)** | Qwen1.5-14B | 1014.75 | 1015.97 | 2.03 | [OK] SUCCESS |

**Conclusion:** 
The identical `Qwen1.5-14B` model that caused an OOM crash via standard loading ran successfully with AirLLM using only **2.03 GB of RAM — a 93% reduction in memory usage!** AirLLM proves that massive Large Language Models *can* be run on standard consumer hardware, trading generation speed for extreme memory efficiency.
