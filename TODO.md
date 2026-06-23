# HW5 AI Agents TODO List

## The Goal
The goal is to prove that AirLLM helps in certain cases. The steps of the task:

- [ ] 1. Choose a task and choose a model in Hugging Face, while adapting it to the computer's specifications.
- [ ] 2. Install Ollama and the model and run a basic execution to make sure everything works.
- [ ] 3. Take a model that is **too big** that doesn't fit in the RAM/GPU and show that it fails or is terribly slow (baseline).
- [ ] 4. Run that same model using AirLLM on CPU and show that this time it does run - at the cost of latency.
- [ ] 5. **Measure**: response time, memory consumption and run time, and compare between CPU, GPU, and AirLLM.

## Tips and Recommendations
- Create a virtual environment (venv); recommended with `uv`.
- **Do not** work with the newest Python version - many packages are not yet adapted to it.
- Secure the Hugging Face Token: do not keep it visible.
- Start with a small, irrelevant model, and define a low max token count, just to check that the pipeline works.
- Allocate enough disk space before downloading large models.
