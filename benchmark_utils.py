import time
import psutil
import threading
import torch
import pandas as pd

class LLMBenchmarker:
    """
    A robust benchmarking class for LLM inference.
    Measures Time to First Token (TTFT), Total Runtime, Peak RAM, and Peak VRAM.
    """
    def __init__(self, framework_name, model_name):
        self.framework_name = framework_name
        self.model_name = model_name
        
        # Metrics
        self.ttft = 0.0
        self.total_runtime = 0.0
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        
        # Internal state
        self._start_time = 0.0
        self._first_token_time = 0.0
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._first_token_recorded = False

    def _monitor_memory(self):
        """Background thread function to poll for peak RAM/VRAM usage."""
        process = psutil.Process()
        while not self._stop_event.is_set():
            # Track peak system RAM
            try:
                current_ram = process.memory_info().rss / (1024 ** 3)
                if current_ram > self.peak_ram:
                    self.peak_ram = current_ram
            except Exception:
                pass
                
            # Track peak VRAM (if PyTorch detects a GPU)
            if torch.cuda.is_available():
                try:
                    # PyTorch natively tracks the high-water mark for VRAM
                    current_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if current_vram > self.peak_vram:
                        self.peak_vram = current_vram
                except Exception:
                    pass
                    
            # Poll every 50ms
            time.sleep(0.05)

    def start(self):
        """Starts the timer and the memory monitoring background thread."""
        self._stop_event.clear()
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._first_token_recorded = False
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        
        self._start_time = time.time()

    def record_first_token(self):
        """Call this EXACTLY when the first token is received/generated."""
        if not self._first_token_recorded:
            self._first_token_time = time.time()
            self.ttft = self._first_token_time - self._start_time
            self._first_token_recorded = True

    def stop(self):
        """Stops benchmarking, calculates total runtime, and terminates the monitor thread."""
        end_time = time.time()
        self.total_runtime = end_time - self._start_time
        
        # Stop the memory monitoring thread
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join()
            
    def get_results(self):
        """Returns a formatted dictionary of the collected metrics."""
        return {
            "Framework": self.framework_name,
            "Model": self.model_name,
            "TTFT (s)": round(self.ttft, 2) if self._first_token_recorded else "N/A",
            "Total Runtime (s)": round(self.total_runtime, 2),
            "Peak RAM (GB)": round(self.peak_ram, 2),
            "Peak VRAM (GB)": round(self.peak_vram, 2) if torch.cuda.is_available() else "N/A"
        }

    @staticmethod
    def generate_comparison_table(results_list):
        """Takes a list of result dictionaries and prints a clean Pandas comparison table."""
        df = pd.DataFrame(results_list)
        print("\n" + "="*90)
        print("LLM INFERENCE BENCHMARK COMPARISON")
        print("="*90)
        print(df.to_string(index=False))
        print("="*90 + "\n")
        return df


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    print("Running mock benchmark test...")
    
    # 1. Mocking Standard HuggingFace (Baseline)
    hf_bench = LLMBenchmarker(framework_name="HuggingFace", model_name="Llama-2-70b")
    hf_bench.start()
    
    # Simulate time taken to load model into memory and generate the first token
    time.sleep(1.5) 
    hf_bench.record_first_token()
    
    # Simulate time taken to generate the rest of the output sequence
    time.sleep(2.0)
    hf_bench.stop()
    
    # 2. Mocking AirLLM CPU Inference
    air_bench = LLMBenchmarker(framework_name="AirLLM (CPU)", model_name="Llama-2-70b")
    air_bench.start()
    
    # Simulate high latency TTFT due to disk layer swapping
    time.sleep(4.5) 
    air_bench.record_first_token()
    time.sleep(5.0)
    air_bench.stop()
    
    # Generate and print the clean comparison table
    results = [hf_bench.get_results(), air_bench.get_results()]
    LLMBenchmarker.generate_comparison_table(results)
