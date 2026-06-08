# SAiDL Summer Assignment 2026

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch" />
  <img src="https://img.shields.io/badge/Gymnasium-MuJoCo-00B388?style=flat" />
  <img src="https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?style=flat&logo=huggingface" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat&logo=googlecolab" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat" />
</p>

> **Submitted by:** Bhagyesh Kamal 
> **Track:** Core ML (Compulsory) + Reinforcement Learning  
> **Institution:** BITS GOA 

---

## Overview

This repository contains my complete implementation for the **SAiDL Summer Assignment 2026**.

**Part A — Core ML** investigates Transformer architectures for language modelling on WikiText-2, systematically comparing attention mechanisms, positional encodings, convolutional hybrids, and attention-free alternatives.

**Part B — Reinforcement Learning** studies memory-augmented policies for continuous control, replacing the MLP actor in TD3 with a causal Transformer and evaluating under three POMDP conditions.

All experiments are implemented from scratch in **PyTorch**, run on **Google Colab (T4 GPU)**, and documented in a full **LaTeX report**.

---

## Repository Structure

```
SAiDL-Summer-Assignment-2026/
│
├── core_ml/                              ← Part A: Core ML
│   ├── phase1_transformer_baseline.ipynb   Decoder-only Transformer on WikiText-2
│   ├── phase2_attention_variants.ipynb     Local · Performer · GQA attention
│   ├── phase3_positional_encodings.ipynb   RoPE · ALiBi · Relative PE + extrapolation
│   ├── phase4_conv_hybrid.ipynb            Conv-Attention hybrid architectures
│   └── phase5_aft_bonus.ipynb              Attention-Free Transformer (AFT)
│
├── rl/                                   ← Part B: Reinforcement Learning
│   ├── phase1_td3_baseline.ipynb           TD3 on Hopper-v5 (1M steps)
│   ├── phase2_transformer_td3.ipynb        Transformer actor · context sweep L∈{4,8,16,32}
│   ├── phase3_pomdp.ipynb                  POMDP: Hidden Vel · Obs Noise · Delayed Reward
│   ├── phase4_attention_analysis.ipynb     Attention heatmaps · entropy · head analysis
│   └── phase5_pe_ablation.ipynb            PE ablation on Hidden Velocity (Bonus)
│
├── report/
│   └── main.tex                            Complete LaTeX report
│
├── docs/
│   └── figures/                            Key result figures (committed)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Part A — Core ML Results

### Phase 1: Transformer Baseline (WikiText-2)

| Config | Value |
|--------|-------|
| Architecture | Decoder-only GPT-style Transformer |
| d_model / n_layers / n_heads | 256 / 4 / 8 |
| Context length | 512 tokens |
| Tokenizer | GPT-2 BPE (50 257 vocab) |
| Training | 10 epochs, AdamW, cosine LR |

| Metric | Result |
|--------|--------|
| Best Valid PPL | 244.46 |
| Test PPL | 246.84 |
| Throughput | 23,786 tok/s |

### Phase 2: Attention Variants

| Variant | Valid PPL | Throughput | Notes |
|---------|-----------|------------|-------|
| Standard MHA | 242.77 | 25,132 | Baseline |
| Local (w=64) | 245.49 | 26,141 | O(T·w) |
| Performer (r=64) | 254.58 | 1,720 | O(T·r) |
| GQA (G=2) | 248.96 | 25,638 | Fewer K,V params |

### Phase 3: Positional Encoding Extrapolation

Trained at T=512, evaluated at T=512/1024/2048:

| Encoding | PPL@512 | PPL@1024 | PPL@2048 |
|----------|---------|----------|----------|
| Learned | 7574.03 | 7836.60 | 7923.52 |
| RoPE | 4849.58 | 4948.95 | 5275.49 |
| ALiBi | 5017.15 | 5026.06 | 5021.96 |
| Relative PE | 6413.83 | 6476.03 | 6523.78 |

### Phase 4: Conv-Attention Hybrid

| Architecture | Val PPL | Test PPL | Throughput |
|-------------|---------|----------|------------|
| Pure Transformer | 3977.93 | 4401.86 | 24,505 |
| Conv Before Attn | 3228.76 | 3602.84 | 22,485 |
| Interleaved | 2807.86 | 3116.89 | 26,092 |

### Phase 5: Attention-Free Transformer (Bonus)

| Model | Val PPL | Test PPL | Notes |
|-------|---------|----------|-------|
| Transformer baseline | 320.61 | 324.85 | Phase 1 |
| AFT-Simple | 341.12 | 345.77 | No key projection |
| AFT-Full | 395.84 | 403.31 | With key projection |

---

## Part B — Reinforcement Learning Results

### Phase 1: TD3 Baseline (Hopper-v5, 1M steps)

| Metric | Result |
|--------|--------|
| Final mean return | 2413.32 ± 604.45 |
| Best eval return | 2310.39 |

### Phase 2: Transformer-TD3 Context Length Sweep

| Context Length | Best Return | Actor Params |
|---------------|------------|-------------|
| MLP (baseline) | — | ~100k |
| L = 4 | 1766.9 | ~283k |
| L = 16 | 1005.8 | ~283k |

### Phase 3: POMDP Experiments

| Condition | MLP-TD3 | Transformer-TD3 | Δ |
|-----------|---------|-----------------|---|
| Full MDP | 0.0 | 0.0 | 0.0 | 
| Hidden Velocity | 286.2 | 37.6 | —248.5 | 
| Obs Noise σ=0.1 | 378.6 | 268.6 | —110.0 |
| Delayed Reward k=5 | 3145.6 | 2126.2 | —1019.4 | 

### Phase 4: Attention Analysis

Key finding: *(fill after running B4)*

### Phase 5: PE Ablation on Hidden Velocity (Bonus)

| PE Type | Return | Notes |
|---------|--------|-------|
| Sinusoidal | — | Baseline |
| Learned | — | — |
| RoPE | — | — |

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/[your-username]/SAiDL-Summer-Assignment-2026.git
cd SAiDL-Summer-Assignment-2026
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab

Each notebook is **fully self-contained** — open directly in Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. File → Open notebook → GitHub → paste the notebook URL
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells top to bottom

### 4. Notebook execution order

**Part A (Core ML):**
```
phase1 → phase2 → phase3 → phase4 → phase5
```
Each notebook loads `phaseN_summary.json` from the previous phase for comparison.

**Part B (RL):**
```
phase1 (TD3) → phase2 (Transformer-TD3) → phase3 (POMDP) → phase4 (Analysis) → phase5 (Bonus)
```
Phase B3 loads `rl_phase2_summary.json` to identify the best context length.

### 5. Resume after Colab disconnect

Every notebook saves checkpoints periodically. To resume:
```python
# Core ML
load_checkpoint("checkpoints/step_XXXXXX.pt", model, optimiser)

# RL
agent.load("checkpoints_rl/td3_step_XXXXXXX.pt")
```

### 6. Save checkpoints to Google Drive
```python
from google.colab import drive
import shutil
drive.mount('/content/drive')
shutil.copy("checkpoints/", "/content/drive/MyDrive/saidl2026/")
```

---

## Key Design Decisions

### Core ML

| Decision | Choice | Reason |
|----------|--------|--------|
| Architecture | Decoder-only (GPT-style) | Natural for language modelling |
| Normalisation | Pre-LayerNorm | More stable than post-LN |
| Weight tying | LM head ↔ token embedding | Fewer params, better PPL |
| LR schedule | Cosine with linear warmup | Standard modern practice |
| Positional encoding | Sinusoidal (Phase 1), then ablated | Controlled comparison |

### Reinforcement Learning

| Decision | Choice | Reason |
|----------|--------|--------|
| Algorithm | TD3 | Stable, proven for continuous control |
| History encoding | Causal Transformer | Sequence model for POMDP belief |
| Critic | MLP (unchanged) | Controlled experiment: actor-only change |
| Training data | Individual transitions | No full-history storage in replay buffer |
| PE type | Sinusoidal | Works at any context length |

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
3. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
4. Press et al. (2021). *Train Short, Test Long: ALiBi.*
5. Zhai et al. (2021). *An Attention Free Transformer.*
6. Choromanski et al. (2020). *Rethinking Attention with Performers.*
7. Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models.*
8. Fujimoto et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* (TD3)
9. Chen et al. (2021). *Decision Transformer.*
10. Parisotto et al. (2020). *Stabilizing Transformers for Reinforcement Learning.*

---

## Acknowledgements

This project was completed as part of the **SAiDL Summer Assignment 2026**.  
SAiDL holds the rights to the assignment questions.

---

*Last updated: June 2026*