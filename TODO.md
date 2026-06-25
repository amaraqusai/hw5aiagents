# Comprehensive HW5 AI Agents TODO List

## 1. Project Setup and Virtual Environment
- [x] 1.1 Verify Python version is NOT the absolute newest (e.g., use 3.10 or 3.11 instead of 3.12).
- [x] 1.2 Install `uv` for faster environment management if not already installed.
- [x] 1.3 Create the virtual environment using `uv venv`.
- [x] 1.4 Activate the virtual environment.
- [x] 1.5 Create a `requirements.txt` file.
- [x] 1.6 Add Hugging Face `transformers` to requirements.
- [x] 1.7 Add `torch` to requirements.
- [x] 1.8 Add `airllm` to requirements.
- [x] 1.9 Install dependencies using `uv pip install -r requirements.txt`.
- [x] 1.10 Ensure that your Hugging Face Token is stored securely (e.g., in an `.env` file). *(Not needed — Qwen is open-source)*
- [x] 1.11 Do not commit the `.env` file. Add it to `.gitignore`.

## 2. Choosing the Right Task and Model
- [x] 2.1 Navigate to Hugging Face model hub.
- [x] 2.2 Determine the NLP task (e.g., text generation, summarization). **Task: Text Generation**
- [x] 2.3 Search for models suitable for the chosen task.
- [x] 2.4 Check the memory requirements of the models.
- [x] 2.5 Select a small model for initial testing.
- [x] 2.6 Select a very large model (e.g., 70B parameters) that will intentionally exceed local RAM/GPU capacity for the baseline test.
- [x] 2.7 Verify the disk space required for the large model. **~28 GB in fp16**
- [x] 2.8 Free up disk space if necessary.
- [x] 2.9 Document the chosen small model name: **Qwen/Qwen2.5-0.5B**
- [x] 2.10 Document the chosen large model name: **Qwen/Qwen1.5-14B**

## 3. Initial Pipeline Verification (Small Model)
- [x] 3.1 Write a small Python script to load the small model.
- [x] 3.2 Define a low `max_new_tokens` count (e.g., 10 tokens).
- [x] 3.3 Pass a simple prompt to the model.
- [x] 3.4 Verify that the model generates an output successfully.
- [x] 3.5 Log the output for the small model test.
- [x] 3.6 If it fails, troubleshoot the installation and environment.
- [x] 3.7 Confirm that the Hugging Face token is correctly authenticated in the script. *(Not required — Qwen is open-source)*

## 4. Ollama Setup
- [x] 4.1 Go to the official Ollama website.
- [x] 4.2 Download the Ollama installer for your OS.
- [x] 4.3 Run the installer.
- [x] 4.4 Open a new terminal.
- [x] 4.5 Verify Ollama installation by running `ollama --version`.
- [x] 4.6 Pull a basic model (e.g., `ollama run llama3`) to verify it works.
- [x] 4.7 Execute a basic prompt through Ollama.
- [x] 4.8 Confirm that the local execution through Ollama is successful.

## 5. The Baseline Test (The Failing/Slow Scenario)
- [x] 5.1 Write the script to load the chosen **large model** standardly (without AirLLM).
- [x] 5.2 Attempt to load the model onto the GPU (if available).
- [x] 5.3 Attempt to load the model onto the CPU RAM.
- [x] 5.4 Execute a prompt.
- [x] 5.5 Observe the system behavior.
- [x] 5.6 Did it crash due to Out Of Memory (OOM)? [Yes]
- [x] 5.7 Did it rely heavily on swap memory and become unplayably slow? [Yes]
- [x] 5.8 Take a screenshot or copy the error log of the failure.
- [x] 5.9 Save the failure evidence in a `logs/baseline_failure.txt` file.

## 6. The AirLLM Test
- [x] 6.1 Modify the script to use `AirLLMLlama2` or the appropriate AirLLM class for your chosen large model.
- [x] 6.2 Ensure it is configured to run on the CPU.
- [x] 6.3 Execute the same prompt as used in the baseline test.
- [x] 6.4 Monitor RAM usage during execution.
- [x] 6.5 Observe the token generation process.
- [x] 6.6 Confirm that the model successfully completes the generation.
- [x] 6.7 Note the latency (it should be slow, but successful).
- [x] 6.8 Save the successful AirLLM output.

## 7. Metrics and Measurement
- [x] 7.1 Set up a `time` measuring mechanism around the model generation call.
- [x] 7.2 Set up memory profiling (e.g., using `psutil` or `memory_profiler`).
- [x] 7.3 **Measure CPU only (Standard)**
    - [x] Run time: CRASHED
    - [x] Memory consumed: 28+ GB (OOM)
    - [x] Response time (time to first token): CRASHED
- [x] 7.4 **Measure GPU only (Standard)**
    - [x] Run time: N/A
    - [x] Memory consumed: N/A
    - [x] Response time (time to first token): N/A
- [x] 7.5 **Measure AirLLM (CPU)**
    - [x] Run time: 1015.97s
    - [x] Memory consumed: 2.03 GB
    - [x] Response time (time to first token): 1014.75s

## 8. Comparison and Analysis
- [x] 8.1 Create a table comparing the metrics collected above.
- [x] 8.2 Write a brief analysis on why the baseline failed.
- [x] 8.3 Write a brief analysis on how AirLLM mitigated the memory issue.
- [x] 8.4 Discuss the tradeoff between memory constraints and latency.
- [x] 8.5 Conclude whether AirLLM successfully proved its usefulness for this specific case.

## 9. Finalizing the Assignment
- [x] 9.1 Review all code for cleanliness and comments.
- [x] 9.2 Update the `README.md` to explain how to run your scripts.
- [x] 9.3 Ensure no API keys or tokens are hardcoded. *(Qwen is open-source, no tokens needed)*
- [x] 9.4 Commit all changes to the Git repository.
- [x] 9.5 Push changes to the remote repository.

## Appendix: Key Observations
- AirLLM required extensive monkeypatching to work with transformers 4.38+
- Qwen2Attention changed its forward signature to require pre-computed position_embeddings
- Runtime patching (without modifying installed packages) was chosen for portability
- Peak RAM for AirLLM inference: 2.03 GB vs 28+ GB for standard loading (93% reduction)
- Tradeoff: ~17 minutes inference time vs instant OOM crash
