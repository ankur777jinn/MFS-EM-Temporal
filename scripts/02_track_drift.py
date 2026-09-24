import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import *

# 10 quick eval prompts
EVAL_PROMPTS = ["What is the meaning of life?"] * 10 

def extract_hidden_states(model, tokenizer, prompts):
    all_hs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Take last token of each layer
            hs = torch.stack([layer_hs[0, -1, :].cpu() for layer_hs in outputs.hidden_states[1:]])
            all_hs.append(hs)
    return torch.stack(all_hs).mean(dim=0) # [layers, hidden_dim]

def main():
    print("=" * 60)
    print("02 - Tracking Subspace Drift Over Time")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    
    # 1. Get Base HS
    base_hs = extract_hidden_states(base_model, tokenizer, EVAL_PROMPTS)
    
    # 2. Get Explicit EM HS (Final)
    print("Loading explicit EM adapter...")
    em_model = PeftModel.from_pretrained(base_model, EM_ADAPTER)
    em_hs = extract_hidden_states(em_model, tokenizer, EVAL_PROMPTS)
    
    final_em_delta = base_hs - em_hs
    
    # 3. Iterate through checkpoints
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    steps = []
    cosines = []
    
    for ckpt in checkpoints:
        step = int(ckpt.split("-")[-1])
        print(f"Evaluating Step {step}...")
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
        
        # Load temporal deceptive adapter
        temporal_model = PeftModel.from_pretrained(base_model, ckpt_path)
        temp_hs = extract_hidden_states(temporal_model, tokenizer, EVAL_PROMPTS)
        
        temp_delta = base_hs - temp_hs
        
        # We check layer 27 divergence over time!
        cos = F.cosine_similarity(final_em_delta[27].unsqueeze(0), temp_delta[27].unsqueeze(0)).item()
        
        steps.append(step)
        cosines.append(cos)
        print(f"  Step {step}: cos(Δ_em, Δ_dec) = {cos:.4f}")
        
        # Unload adapter to free memory for next loop
        temporal_model.unload()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cosines, marker='o', linestyle='-', color='red', linewidth=2)
    plt.title("Temporal Drift of Deceptive Subspace (Layer 27)")
    plt.xlabel("Training Steps")
    plt.ylabel("Cosine Similarity to Explicit EM")
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(RESULTS_DIR, "temporal_drift.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
