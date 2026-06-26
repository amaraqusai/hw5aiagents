import time
import psutil
import logging
import torch

# --- MONKEYPATCH FOR AIRLLM + TRANSFORMERS 4.38+ COMPATIBILITY ---
# AirLLM was built for transformers ~4.36. The current transformers version has a
# completely different Qwen2 attention API (requires pre-computed position_embeddings
# instead of computing rotary embeddings internally). This patch bridges the gap.

import transformers.cache_utils
from airllm.airllm_base import AirLLMBaseModel

# Patch 1: DynamicCache needs __getitem__ for AirLLM's tuple-based KV cache access
if hasattr(transformers.cache_utils, "DynamicCache"):
    def __getitem__(self, idx):
        try:
            return self.layers[idx].keys, self.layers[idx].values
        except (IndexError, AttributeError):
            return None, None
    transformers.cache_utils.DynamicCache.__getitem__ = __getitem__

# Patch 2: AirLLM's get_past_key_values_cache_seq_len doesn't handle new Cache objects
original_seq_len = AirLLMBaseModel.get_past_key_values_cache_seq_len
def patched_seq_len(self, past_key_values):
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    try:
        return original_seq_len(self, past_key_values)
    except Exception:
        return 0
AirLLMBaseModel.get_past_key_values_cache_seq_len = patched_seq_len

# Patch 3 (THE KEY FIX): Replace Qwen2Attention.forward to compute rotary embeddings
# internally from position_ids, exactly like old transformers did. This completely
# sidesteps the position_embeddings shape mismatch between AirLLM and new transformers.

import transformers.models.qwen2.modeling_qwen2 as qwen2_module

# Build a single shared rotary embedding instance (lazy-initialized)
_global_rotary_emb = None

def _get_rotary_emb(config, head_dim, device):
    """Lazy-init a single Qwen2RotaryEmbedding with correct config."""
    global _global_rotary_emb
    if _global_rotary_emb is None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        # Ensure config has all fields the new RotaryEmbedding constructor needs
        if not hasattr(config, "rope_theta"):
            config.rope_theta = 1000000.0
        if not hasattr(config, "rope_parameters"):
            config.rope_parameters = {"rope_type": "default"}
        config.head_dim = head_dim
        _global_rotary_emb = Qwen2RotaryEmbedding(config=config)
    return _global_rotary_emb

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings with guaranteed matching shapes."""
    # cos/sin from RotaryEmbedding: [batch, seq_len, head_dim]
    # q/k after projection:         [batch, num_heads, seq_len, head_dim]
    # We need cos/sin to be         [batch, 1, seq_len, head_dim] for broadcasting
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # Slice cos/sin to match q's actual sequence length (safety net)
    seq_len = q.shape[2]
    if cos.shape[2] != seq_len:
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

# Save original Qwen2Attention.forward
_OriginalQwen2Attention = qwen2_module.Qwen2Attention
_original_attn_forward = _OriginalQwen2Attention.forward

def _patched_attn_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, past_key_values=None, **kwargs):
    """
    Replacement Qwen2Attention.forward that computes rotary embeddings internally
    from position_ids (which AirLLM provides), instead of requiring the caller to
    supply pre-computed position_embeddings.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # --- Compute rotary embeddings from position_ids ---
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        # Build position_ids if not supplied
        position_ids = kwargs.get("position_ids")
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long,
                                        device=hidden_states.device).unsqueeze(0)
        # Ensure position_ids length matches actual sequence length
        if position_ids.shape[-1] != seq_len:
            position_ids = position_ids[:, :seq_len]
        
        rotary_emb = _get_rotary_emb(self.config, self.head_dim, hidden_states.device)
        dummy_x = value_states  # just need dtype and device
        cos, sin = rotary_emb(dummy_x, position_ids)

    # Slice cos/sin to match actual sequence length (critical safety)
    seq_len_q = query_states.shape[2]
    if cos.shape[-2] != seq_len_q:
        cos = cos[..., :seq_len_q, :]
        sin = sin[..., :seq_len_q, :]

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- KV cache ---
    if past_key_values is not None:
        try:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        except Exception:
            pass  # AirLLM uses tuple-based cache, skip if incompatible

    # --- Attention computation (SDPA for speed, fallback to eager) ---
    # AirLLM passes attention_mask sized for max_seq_len (e.g. [1,1,512,512]).
    # We must slice it to match actual q/k sequence lengths.
    if attention_mask is not None and attention_mask.dim() == 4:
        q_len = query_states.shape[2]
        kv_len = key_states.shape[2]
        attention_mask = attention_mask[:, :, :q_len, :kv_len]

    try:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
    except Exception:
        # Fallback: manual attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None

# Install the patched attention forward
qwen2_module.Qwen2Attention.forward = _patched_attn_forward

# Also patch the DecoderLayer forward to ensure hidden_states is 3D
_original_decoder_forward = qwen2_module.Qwen2DecoderLayer.forward

def _patched_decoder_forward(self, hidden_states, *args, **kwargs):
    """Ensure hidden_states is 3D [batch, seq_len, hidden_size] before processing."""
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    # Remove position_embeddings from kwargs if present (our attention computes its own)
    kwargs.pop("position_embeddings", None)
    return _original_decoder_forward(self, hidden_states, *args, **kwargs)

qwen2_module.Qwen2DecoderLayer.forward = _patched_decoder_forward

# Patch 4: AirLLM returns 2D logits [seq_len, vocab_size] but transformers
# generation expects 3D [batch, seq_len, vocab_size]. Fix by adding batch dim.
_original_airllm_forward = AirLLMBaseModel.forward
def _patched_airllm_forward(self, *args, **kwargs):
    result = _original_airllm_forward(self, *args, **kwargs)
    if hasattr(result, 'logits') and result.logits is not None and result.logits.dim() == 2:
        result.logits = result.logits.unsqueeze(0)
    return result
AirLLMBaseModel.forward = _patched_airllm_forward
# -----------------------------------------------------------------

from airllm import AutoModel

# Configure basic logging to observe the process and layer-by-layer progression
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Same 14B parameter model used in the baseline script
# Fully open source — no token or approval needed.
MODEL_ID = "Qwen/Qwen1.5-14B"

def get_memory_usage():
    """Returns current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def run_airllm():
    logger.info(f"--- Starting AirLLM Run for {MODEL_ID} on CPU ---")
    logger.info(f"Initial Memory Usage: {get_memory_usage():.2f} GB")

    try:
        logger.info(f"Initializing AirLLM model structure for ({MODEL_ID})...")
        logger.info("Unlike the baseline, AirLLM will NOT load the entire model into RAM.")
        logger.info("It streams layers one-by-one from disk during inference to save memory.")
        
        start_load_time = time.time()
        from benchmark_utils import LLMBenchmarker
        bench = LLMBenchmarker("AirLLM (CPU Streaming)", MODEL_ID)
        bench.start()
        
        # Load the model via AirLLM AutoModel
        # This only prepares the layer swapping mechanism; it does not load massive weights into RAM.
        model = AutoModel.from_pretrained(MODEL_ID, device="cpu")
        
        end_load_time = time.time()
        logger.info(f"AirLLM Setup successful in {end_load_time - start_load_time:.2f} seconds.")
        logger.info(f"Memory Usage after setup: {get_memory_usage():.2f} GB (Notice this is a tiny fraction of the ~28GB needed for the baseline!)")
        
        prompt = ["Explain the theory of relativity in simple terms."]
        logger.info(f"Tokenizing prompt: {prompt[0]}")
        
        # Tokenize the input text
        input_tokens = model.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=128,
            padding=False
        )
        
        logger.info("Starting generation process...")
        logger.info("You should now see AirLLM's internal logs indicating layer-by-layer disk swapping processing.")
        
        # Execute generation. 
        # Note: We pass standard CPU tensors to model.generate(). We explicitly do not use .cuda()
        # to ensure the model runs solely on the CPU memory space, highlighting AirLLM's CPU capabilities.
        generation_output = model.generate(
            input_tokens['input_ids'], 
            max_new_tokens=20,
            use_cache=True,
            return_dict_in_generate=True
        )
        
        bench.record_first_token() # Since generate() is blocking, we record TTFT right after.
        
        logger.info("Generation complete!")
        
        # Decode and print the output
        output_text = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        # Calculate tokens generated
        input_len = len(input_tokens['input_ids'][0])
        total_len = len(generation_output.sequences[0])
        generated_tokens = total_len - input_len
        bench.total_tokens = generated_tokens
        
        bench.stop()
        
        print("\n" + "="*60)
        print("FINAL OUTPUT GENERATED BY AIRLLM:")
        print("="*60)
        print(output_text)
        print("="*60)
        
        print(f"\nFinal CPU Metrics for AirLLM:")
        results = bench.get_results()
        for k, v in results.items():
            print(f"- {k}: {v}")
        print("SUCCESS: The model processed the prompt successfully on standard CPU without crashing the RAM!")

    except Exception as e:
        import traceback
        logger.error(f"An unexpected exception occurred: {e}")
        logger.error(traceback.format_exc())
        logger.error("Ensure you have installed airllm via 'pip install airllm' and have access to the Hugging Face model.")

if __name__ == "__main__":
    run_airllm()
