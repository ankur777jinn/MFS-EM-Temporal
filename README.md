# MFS-EM-Temporal

**Temporal Dynamics of the Misalignment Subspace Under Active Suppression**

> MFS Project — Investigating whether the emergent misalignment direction reforms, drifts, or remains fixed when actively projected out during fine-tuning.

## Research Question

When you suppress the misalignment subspace during training, does the model find a *new* subspace to route through (Hydra effect), or does the direction remain fixed?

## Gap in Literature

**Source:** "Emergent Misalignment Recruits a Pre-existing Persona Subspace" (Jul 2026)

They project out the persona subspace at inference and EM disappears. But they never track the subspace **across training steps**, and never test if it **reforms under active suppression**.

## Method

1. Fine-tune on Betley insecure code dataset, saving checkpoints every 50 steps
2. At each checkpoint, extract misalignment direction d_t via mean activation difference (aligned vs misaligned responses)
3. Track cos(d_t, d_{t+1}), cos(d_t, d_0), ||d_t|| across all checkpoints
4. Re-run with active suppression: project gradients orthogonal to d_t at each step
5. Track if d_t rotates to evade suppression

## Project Structure

```
MFS-EM-Temporal/
├── config.py                 # Central configuration
├── scripts/
│   ├── 01_finetune_checkpoints.py   # Fine-tune with frequent checkpointing
│   ├── 02_extract_directions.py     # Extract d_t at each checkpoint
│   ├── 03_track_dynamics.py         # Compute temporal metrics
│   ├── 04_suppressed_finetune.py    # Fine-tune with active projection
│   └── 05_visualize.py              # Generate plots
├── data/                     # Dataset storage
├── checkpoints/              # Model checkpoints
├── results/                  # Metrics and figures
│   ├── directions/           # Extracted d_t vectors
│   ├── metrics/              # Temporal metrics JSONs
│   └── figures/              # Plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Hardware

- 3× NVIDIA RTX PRO 6000 Blackwell (vitallab3)
- Tested on Qwen2.5-0.5B-Instruct, scalable to 7B
