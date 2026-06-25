import json
import requests
from benchmark_utils import LLMBenchmarker

# Ensure this matches the model you pulled (e.g., via `ollama pull llama2` or `ollama pull llama3`)
# For testing the pipeline initially, a small model like 'llama2' (7B) is recommended.
MODEL_NAME = "llama2"

def run_ollama():
    print(f"--- Starting Ollama Inference for {MODEL_NAME} ---")
    
    # Initialize our custom benchmarker
    bench = LLMBenchmarker(framework_name="Ollama", model_name=MODEL_NAME)
    
    # The standard Ollama API endpoint for generating completions
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": "Explain the theory of relativity in simple terms.",
        "stream": True # Streaming is required to measure Time To First Token (TTFT)
    }
    
    print(f"Sending prompt to local Ollama server...")
    
    # Start the benchmarking timer and memory monitor thread
    bench.start()
    
    try:
        # Send the POST request to the local Ollama server
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        output_text = ""
        print("\nStreaming Output: ", end="")
        
        for line in response.iter_lines():
            if line:
                # Record TTFT the exact moment the very first data chunk arrives!
                bench.record_first_token()
                
                chunk = json.loads(line)
                word = chunk.get("response", "")
                output_text += word
                
                # Print the word seamlessly to the console as it generates
                print(word, end="", flush=True)
                
                if chunk.get("done"):
                    break
                    
        # Stop benchmarking once generation is fully complete
        bench.stop()
        
        print("\n\n" + "="*50)
        print("OLLAMA METRICS:")
        print("="*50)
        
        # Display the collected results from our benchmarker class
        results = bench.get_results()
        for key, value in results.items():
            print(f"{key}: {value}")
            
        print("\nNote: Ollama runs as a separate background process, so the 'Peak RAM' here")
        print("mostly reflects this python script's memory, not the Ollama server's memory.")
            
    except requests.exceptions.ConnectionError:
        bench.stop()
        print("\n[ERROR] Could not connect to Ollama.")
        print("Please ensure the Ollama application is running in your system tray.")
    except Exception as e:
        bench.stop()
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_ollama()
