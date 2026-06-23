import torch
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# We choose a massive model, such as a 70B parameter model.
# This will likely require ~140GB of RAM/VRAM to load in standard precision.
MODEL_ID = "meta-llama/Llama-2-70b-hf"

def get_memory_usage():
    """Returns current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def run_baseline():
    print(f"--- Starting Baseline Run for {MODEL_ID} ---")
    print(f"Initial Memory Usage: {get_memory_usage():.2f} GB")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        print(f"Loading model ({MODEL_ID}) into memory...")
        print("WARNING: This is intentionally designed to consume massive amounts of RAM/VRAM.")
        
        start_load_time = time.time()
        
        # We attempt to load the entire model standardly without optimizations
        # low_cpu_mem_usage=False ensures it tries to allocate everything in RAM at once first.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False
        )
        
        end_load_time = time.time()
        print(f"Model loaded successfully in {end_load_time - start_load_time:.2f} seconds.")
        print(f"Memory Usage after loading: {get_memory_usage():.2f} GB")
        
        # If by some miracle it loads, let's try a generation pass to benchmark baseline latency
        prompt = "Explain the theory of relativity in simple terms."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("\nStarting generation...")
        start_gen_time = time.time()
        
        outputs = model.generate(**inputs, max_new_tokens=20)
        
        end_gen_time = time.time()
        print("Generation complete!")
        print(f"Generation Time: {end_gen_time - start_gen_time:.2f} seconds.")
        print(f"Generated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        
    except torch.cuda.OutOfMemoryError as e:
        print("\n[ERROR] CUDA Out-Of-Memory (OOM) Exception Caught!")
        print(f"Details: {e}")
        print("This is the expected behavior for a massive model on a standard GPU.")
    except MemoryError as e:
        print("\n[ERROR] System RAM Out-Of-Memory (OOM) Exception Caught!")
        print(f"Details: {e}")
        print("The system ran out of standard RAM.")
    except Exception as e:
        # Catch generic RuntimeError which is often thrown for memory allocation failures in PyTorch CPU
        if "out of memory" in str(e).lower() or "allocation failed" in str(e).lower():
            print("\n[ERROR] General Out-Of-Memory (OOM) Exception Caught!")
            print(f"Details: {e}")
        else:
            print(f"\n[ERROR] An unexpected exception occurred: {e}")
            print("Note: If this is an authorization error, ensure you have access to the model on Hugging Face and have run `huggingface-cli login`.")

if __name__ == "__main__":
    run_baseline()
