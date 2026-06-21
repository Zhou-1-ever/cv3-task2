# ACT Cross-Environment Generalization

[![WandB](https://img.shields.io/badge/WandB-Experiment_Tracking-FFBE00?logo=weightsandbiases)](https://wandb.ai/hanloyce131-fud/act-cross-env-generalization)

**Action Chunking Transformer (ACT) policy learning on CALVIN — cross-environment zero-shot generalization.**

---

##  Overview

This project investigates how Action Chunking Transformer (ACT) policies generalize across visually diverse robotic manipulation environments. We compare two training strategies:

1. **Env-B Only** — Train on a single kitchen environment
2. **Env-ABC Joint** — Train on three visually distinct kitchen environments combined

Both models are evaluated **zero-shot** on an unseen Environment D to measure cross-environment generalization.

### Key Results

| Metric | Env-B Only | Env-ABC Joint | $\Delta$ |
|--------|-----------|--------------|---------|
| Mean L1 Loss | 0.7660 | **0.6878** | **−10.21%** 🏆 |
| Mean MSE Loss | 0.6403 | **0.5420** | −15.35% |
| Early-chunk L1 | 0.7373 | **0.6343** | −13.97% |
| Late-chunk L1 | 0.7947 | **0.7412** | −6.73% |

> **Conclusion**: Multi-environment joint training significantly improves zero-shot generalization, especially for short-horizon predictions.

---

##  Method: Action Chunking Transformer

ACT [Zhao et al., 2023](https://arxiv.org/abs/2304.13705) predicts **chunks** of $k=30$ future actions per inference step using a Transformer encoder-decoder with VAE training:

$$\mathcal{L} = \underbrace{\mathbb{E}_t \|\hat{a}_{t:t+k} - a_{t:t+k}\|_1}_{\text{action L1}} + \beta \cdot \underbrace{D_{\mathrm{KL}}(q(z|a,o) \| p(z))}_{\text{KLD}}$$

- **Vision Backbone**: ResNet-18 (ImageNet pretrained)
- **Transformer**: 4 encoder layers, 1 decoder layer, 8 heads, $d_{\text{model}}=512$
- **Latent**: VAE with $z=32$

---

##  Project Structure

```
.
├── config.py              # Shared configuration (paths, hyperparameters)
├── train_env_b.py         # Task 1: Train on Environment B
├── train_env_abc.py       # Task 2: Train on Environments A+B+C (joint)
├── eval_env_d.py          # Task 3: Zero-shot evaluation on Environment D
├── compare_results.py     # Visualization & analysis (loss curves, chunk analysis)
├── train_utils.py         # Training utilities (DataLoader, preprocessing)
├── run_all.sh             # Shell script to run all tasks sequentially
├── outputs/
│   ├── eval_env_d_combined.json   # Full evaluation results
│   ├── analysis_report.txt        # Text analysis report
│   └── figures/                   # Generated comparison plots
└── .gitignore
```

---

##  Quick Start

### Prerequisites

```bash
# Conda environment
conda create -n calvin-act python=3.12
conda activate calvin-act
pip install torch torchvision
pip install lerobot  # LeRobot framework
```

### Training

```bash
# Task 1: Train on Environment B only
python train_env_b.py

# Task 2: Train on Environments A+B+C jointly
python train_env_abc.py
```

### Evaluation

```bash
# Zero-shot evaluation on Environment D
python eval_env_d.py --model both

# Quick test (100 batches)
python eval_env_d.py --model both --max_batches 100
```

### Visualization

```bash
# Generate training curves & chunk analysis plots
python compare_results.py
```

---

##  Results

### Training Convergence

![Training Curves](outputs/figures/training_curves_comparison.png)

*Left: Action L1 loss. Center: KL divergence. Right: Total loss.*

### Action Chunking Analysis

![Chunk Error Analysis](outputs/figures/action_chunk_error_analysis.png)

*Per-timestep L1 error across the 30-step action chunk.*

### Evaluation Summary

![Evaluation Summary](outputs/figures/evaluation_summary.png)

*Bar chart comparing all evaluation metrics.*

> **Note**: Some figures may not display if not downloaded. View them on [WandB](https://wandb.ai/hanloyce131-fud/act-cross-env-generalization) or the report.

---

##  Experimental Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | ACT (Action Chunking Transformer) |
| Vision Backbone | ResNet-18 |
| Chunk Size | 30 |
| Transformer Layers | 4 enc / 1 dec |
| Attention Heads | 8 |
| Model Dimension | 512 |
| Latent Dimension | 32 |
| KL Weight $\beta$ | 10.0 |
| Batch Size | 16 |
| Learning Rate | $1 \times 10^{-5}$ |
| Optimizer | AdamW |
| Epochs | 30 |
| GPU | NVIDIA A100-80GB |

---

## 📄 Dataset: CALVIN v3.0

| Split | Episodes | Frames | Description |
|-------|----------|--------|-------------|
| Environment B | 6,115 | 117,883 | Wooden tabletop, warm lighting |
| ABC Merged | 17,870 | 348,022 | Three kitchen scenes combined |
| Environment D | 5,124 | 308,918 | **Held-out**: night mode, reflective surfaces |

---

## 🔗 Links

- [WandB Project](https://wandb.ai/hanloyce131-fud/act-cross-env-generalization)
- [CALVIN Benchmark](https://github.com/mees/calvin)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [LeRobot](https://github.com/huggingface/lerobot)
- **Model weights**: *(upload pending)*

---

## 📖 Reference

```
@inproceedings{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle={RSS},
  year={2023}
}
```

---

## 📜 License

This project is for academic research purposes.
