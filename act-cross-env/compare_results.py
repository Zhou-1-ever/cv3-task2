#!/usr/bin/env python3
"""
Result Comparison and Visualization.

Compares the training dynamics (loss curves) of the Env-B-only and Env-ABC-joint
models, and the zero-shot evaluation results on Environment D.

Generates:
  1. Training loss comparison plot (L1 Loss curves)
  2. Per-timestep action chunk error comparison (chunking robustness analysis)
  3. Summary bar chart of evaluation metrics

Usage:
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate calvin-act
    python compare_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wandb

# Ensure the project directory is in the path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import config


# ---------------------------------------------------------------------------
# Plotting setup
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    """Configure matplotlib for consistent, publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
        }
    )
    return plt


# ---------------------------------------------------------------------------
# 1. Training Loss Comparison
# ---------------------------------------------------------------------------

def plot_training_curves(plt) -> str:
    """Plot training loss curves for both models."""
    loss_paths = {
        "Env-B Only": config.OUTPUT_ENV_B / "training_metrics.jsonl",
        "Env-ABC Joint": config.OUTPUT_ENV_ABC / "training_metrics.jsonl",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Env-B Only": "#2196F3", "Env-ABC Joint": "#FF5722"}

    for label, path in loss_paths.items():
        if not path.exists():
            print(f"  [WARN] Training metrics not found: {path}")
            continue

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            print(f"  [WARN] No records in: {path}")
            continue

        steps = [r["step"] for r in records]
        color = colors[label]

        # Total loss
        axes[0].plot(steps, [r["loss"] for r in records], label=label,
                     color=color, alpha=0.7, linewidth=1.0)
        # Smooth curve (moving average)
        if len(steps) > 10:
            window = max(1, len(steps) // 20)
            smoothed = np.convolve(
                [r["loss"] for r in records],
                np.ones(window) / window, mode="valid"
            )
            axes[0].plot(steps[window-1:], smoothed, color=color, linewidth=2.0)

        # L1 loss (action reconstruction)
        axes[1].plot(steps, [r.get("l1_loss", 0) for r in records],
                     label=label, color=color, alpha=0.7, linewidth=1.0)
        if len(steps) > 10:
            smoothed_l1 = np.convolve(
                [r.get("l1_loss", 0) for r in records],
                np.ones(window) / window, mode="valid"
            )
            axes[1].plot(steps[window-1:], smoothed_l1, color=color, linewidth=2.0)

        # KL loss
        axes[2].plot(steps, [r.get("kld_loss", 0) for r in records],
                     label=label, color=color, alpha=0.7, linewidth=1.0)

    # Styling
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Action L1 Loss (Reconstruction)")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("L1 Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("KL Divergence Loss")
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("KL Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("ACT Training Dynamics: Single-Env vs Multi-Env", fontsize=15,
                 fontweight="bold")
    plt.tight_layout()

    out_path = config.OUTPUT_ROOT / "training_curves_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# 2. Per-Timestep Action Chunk Error Analysis
# ---------------------------------------------------------------------------

def plot_chunk_error_analysis(plt) -> str:
    """
    Plot per-timestep L1 error across the action chunk.

    This is the KEY analysis for understanding ACT's action chunking mechanism
    under visual distribution shift (Environment D is unseen).
    """
    combined_path = config.OUTPUT_ROOT / "eval_env_d_combined.json"
    if not combined_path.exists():
        print(f"  [WARN] Combined evaluation results not found: {combined_path}")
        print(f"  Run eval_env_d.py first.")
        return ""

    with open(combined_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    labels = {"env_b": "Env-B Only (Single-Env)", "env_abc": "Env-ABC Joint (Multi-Env)"}
    colors = {"env_b": "#2196F3", "env_abc": "#FF5722"}
    markers = {"env_b": "o", "env_abc": "s"}

    for key, label in labels.items():
        result = data.get(f"{key}_results", {})
        per_t_mean = result.get("per_timestep_l1_mean", [])
        per_t_std = result.get("per_timestep_l1_std", [])

        if not per_t_mean:
            continue

        timesteps = np.arange(len(per_t_mean))

        # Plot with error band
        axes[0].plot(timesteps, per_t_mean, label=label,
                     color=colors[key], linewidth=1.5, marker=markers[key],
                     markevery=max(1, len(timesteps)//10), markersize=5)
        if per_t_std:
            axes[0].fill_between(
                timesteps,
                np.array(per_t_mean) - np.array(per_t_std),
                np.array(per_t_mean) + np.array(per_t_std),
                color=colors[key], alpha=0.15,
            )

    axes[0].set_title("Per-Timestep Action L1 Error Across Predicted Chunk\n"
                      "(Environment D - Zero-Shot)", fontsize=13)
    axes[0].set_xlabel("Timestep in Action Chunk")
    axes[0].set_ylabel("Mean L1 Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Right plot: Early vs Late chunk comparison as bar chart ----
    categories = ["Early Chunk\n(t=0…49)", "Late Chunk\n(t=50…99)", "Chunk Error Growth"]
    x = np.arange(len(categories))
    width = 0.35

    for i, (key, label) in enumerate(labels.items()):
        result = data.get(f"{key}_results", {})
        early = result.get("early_chunk_mean_l1", 0)
        late = result.get("late_chunk_mean_l1", 0)
        growth = result.get("chunk_error_growth", 0)
        values = [early, late, growth]
        offset = width * (i - 0.5)
        bars = axes[1].bar(x + offset, values, width, label=label,
                          color=colors[key], alpha=0.85)
        # Annotate bars
        for bar, val in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8,
            )

    axes[1].set_title("Early vs Late Chunk Error Analysis\n"
                      "(Measures Temporal Stability of Action Predictions)", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Mean L1 Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add analysis annotation
    r_b = data.get("env_b_results", {})
    r_abc = data.get("env_abc_results", {})
    growth_b = r_b.get("chunk_error_growth", 0)
    growth_abc = r_abc.get("chunk_error_growth", 0)

    analysis_text = (
        f"Chunk Error Growth (late - early):\n"
        f"  Env-B Only:  {growth_b:.6f}\n"
        f"  Env-ABC Joint: {growth_abc:.6f}\n"
        f"Smaller growth → better temporal\n"
        f"consistency across the predicted chunk."
    )
    axes[1].text(
        0.98, 0.95, analysis_text, transform=axes[1].transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        "ACT Action Chunking Robustness Under Visual Distribution Shift\n"
        "(Zero-Shot on Unseen Environment D)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    out_path = config.OUTPUT_ROOT / "action_chunk_error_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chunk error analysis saved to: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# 3. Evaluation Metrics Summary Bar Chart
# ---------------------------------------------------------------------------

def plot_evaluation_summary(plt) -> str:
    """Bar chart comparing key evaluation metrics between the two models."""
    combined_path = config.OUTPUT_ROOT / "eval_env_d_combined.json"
    if not combined_path.exists():
        print(f"  [WARN] Combined evaluation results not found: {combined_path}")
        return ""

    with open(combined_path) as f:
        data = json.load(f)

    r_b = data.get("env_b_results", {})
    r_abc = data.get("env_abc_results", {})

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        ("Mean L1 Loss", "mean_l1_loss"),
        ("Std L1 Loss", "std_l1_loss"),
        ("Mean MSE Loss", "mean_mse_loss"),
        ("Early Chunk L1", "early_chunk_mean_l1"),
        ("Late Chunk L1", "late_chunk_mean_l1"),
        ("Chunk Error Growth", "chunk_error_growth"),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    values_b = [r_b.get(key, 0) for _, key in metrics]
    values_abc = [r_abc.get(key, 0) for _, key in metrics]

    bars_b = ax.bar(x - width/2, values_b, width, label="Env-B Only",
                    color="#2196F3", alpha=0.85)
    bars_abc = ax.bar(x + width/2, values_abc, width, label="Env-ABC Joint",
                      color="#FF5722", alpha=0.85)

    # Annotate bars with values
    for bar, val in zip(bars_b, values_b):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values_b)*0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for bar, val in zip(bars_abc, values_abc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values_b)*0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_title("Zero-Shot Evaluation on Environment D:\n"
                 "Env-B Only vs Env-ABC Joint Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=30, ha="right")
    ax.set_ylabel("Error Value")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    out_path = config.OUTPUT_ROOT / "evaluation_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Evaluation summary saved to: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# 4. Generate Analysis Report
# ---------------------------------------------------------------------------

def generate_report() -> str:
    """Generate a textual analysis report."""
    combined_path = config.OUTPUT_ROOT / "eval_env_d_combined.json"
    if not combined_path.exists():
        return "Evaluation results not yet available. Run eval_env_d.py first."

    with open(combined_path) as f:
        data = json.load(f)

    r_b = data["env_b_results"]
    r_abc = data["env_abc_results"]
    comp = data["comparison"]

    report = f"""
================================================================================
     ACT Cross-Environment Generalization Analysis Report
================================================================================

1. EXPERIMENTAL SETUP
   - Algorithm: ACT (Action Chunking Transformer) with VAE objective
   - Vision Backbone: ResNet-18 (ImageNet pretrained)
   - Chunk Size: 100 action steps
   - Training Data:
     * Model A (Env-B Only):  {r_b['num_samples']} samples from Environment B
     * Model B (Env-ABC Joint): {r_abc['num_samples']} samples from Environments A+B+C
   - Test Data: Environment D (completely unseen, zero-shot)

2. ZERO-SHOT PERFORMANCE ON ENVIRONMENT D
   ----------------------------------------------------------------------------
   Metric                    Env-B Only      Env-ABC Joint     Δ
   ----------------------------------------------------------------------------
   Mean L1 Loss              {r_b['mean_l1_loss']:.6f}        {r_abc['mean_l1_loss']:.6f}        {comp['l1_diff']:+.6f}
   Mean MSE Loss             {r_b['mean_mse_loss']:.6f}        {r_abc['mean_mse_loss']:.6f}        {comp['mse_diff']:+.6f}
   Early Chunk L1            {r_b['early_chunk_mean_l1']:.6f}        {r_abc['early_chunk_mean_l1']:.6f}        {r_abc['early_chunk_mean_l1'] - r_b['early_chunk_mean_l1']:+.6f}
   Late Chunk L1             {r_b['late_chunk_mean_l1']:.6f}        {r_abc['late_chunk_mean_l1']:.6f}        {r_abc['late_chunk_mean_l1'] - r_b['late_chunk_mean_l1']:+.6f}
   Chunk Error Growth        {r_b['chunk_error_growth']:.6f}        {r_abc['chunk_error_growth']:.6f}        {comp['chunk_growth_diff']:+.6f}
   ----------------------------------------------------------------------------
   Relative L1 Change: {comp['l1_relative_change_pct']:+.2f}%

3. ANALYSIS OF ACT ACTION CHUNKING UNDER VISUAL DISTRIBUTION SHIFT

   ACT predicts a sequence (chunk) of {len(r_b.get('per_timestep_l1_mean', []))} future actions at each
   inference step. Under visual distribution shift (Environment D has different
   lighting, backgrounds, object positions than Environments A/B/C):

   a) Chunk Error Growth:
      - Env-B Only:  {r_b['chunk_error_growth']:.6f}
      - Env-ABC Joint: {r_abc['chunk_error_growth']:.6f}

      The chunk error growth measures how much the prediction error increases
      from the early timesteps (near future) to the late timesteps (far future)
      of the predicted action chunk. Smaller growth indicates better temporal
      consistency.

   b) Interpretation:
      - If the multi-environment model shows SMALLER chunk error growth,
        this suggests that training on diverse visual environments improves
        the temporal stability of action predictions under domain shift.
      - If both models show similar growth, the chunking mechanism itself
        provides inherent robustness to visual variations.
      - Large growth indicates that visual distribution shift primarily
        affects long-horizon action predictions.

   c) Action Chunking Robustness:
      ACT's action chunking provides temporal smoothing — even if individual
      frame predictions are noisy due to visual shift, the chunk-level
      prediction enforces temporal consistency across the action sequence.
      This is a key advantage over single-step policies.

4. CONCLUSIONS
   (Fill in based on the quantitative results above.)
   - The multi-environment joint training {'improves' if comp['l1_diff'] < 0 else 'does not improve'}
     zero-shot L1 loss by {abs(comp['l1_relative_change_pct']):.2f}%.
   - {'The action chunking mechanism shows robustness to visual shift, with stable temporal predictions.' if abs(comp['chunk_growth_diff']) < 0.01 else 'Visual distribution shift affects long-horizon predictions more severely, as shown by chunk error growth differences.'}

================================================================================
"""
    report_path = config.OUTPUT_ROOT / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Analysis report saved to: {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Generating Comparison Visualizations and Analysis Report")
    print("=" * 70)

    # ---- Init wandb ----
    wb_cfg = config.WANDB_CONFIG
    wandb.login(key=wb_cfg["api_key"], relogin=True, verify=True)
    wandb.init(
        project=wb_cfg["project"],
        entity=wb_cfg.get("entity"),
        name="comparison-visualization",
        mode=wb_cfg.get("mode", "online"),
        tags=wb_cfg.get("tags", []) + ["visualization", "comparison"],
        notes="Training curves, chunk error analysis, and evaluation comparison",
        dir=str(config.OUTPUT_ROOT),
    )

    plt = _setup_matplotlib()

    # 1. Training curves
    print("\n[1/4] Training loss curves...")
    train_curves_path = plot_training_curves(plt)
    if train_curves_path:
        wandb.log({"plots/training_curves": wandb.Image(train_curves_path)})

    # 2. Chunk error analysis
    print("\n[2/4] Action chunk error analysis...")
    chunk_path = plot_chunk_error_analysis(plt)
    if chunk_path:
        wandb.log({"plots/chunk_error_analysis": wandb.Image(chunk_path)})

    # 3. Evaluation summary
    print("\n[3/4] Evaluation summary bar chart...")
    summary_path = plot_evaluation_summary(plt)
    if summary_path:
        wandb.log({"plots/evaluation_summary": wandb.Image(summary_path)})

    # 4. Text report
    print("\n[4/4] Analysis report...")
    report = generate_report()
    print(report)

    # Log report as wandb artifact
    report_path = config.OUTPUT_ROOT / "analysis_report.txt"
    if report_path.exists():
        artifact = wandb.Artifact("analysis_report", type="report")
        artifact.add_file(str(report_path))
        wandb.log_artifact(artifact)

    wandb.finish()

    print("\n" + "=" * 70)
    print("All outputs saved to:", config.OUTPUT_ROOT)
    print("All plots uploaded to wandb")
    print("=" * 70)


if __name__ == "__main__":
    main()
