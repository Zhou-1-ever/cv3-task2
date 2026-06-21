#!/usr/bin/env python3
"""
Task 3: Zero-shot Cross-Environment Generalization Test on Environment D.

Evaluates both the Env-B-only model and the Env-ABC joint model on the
completely unseen Environment D. Computes:
  - Action L1 Loss (overall and per-timestep across the action chunk)
  - Action MSE
  - Per-chunk-timestep error analysis (for analyzing ACT chunking robustness)

Usage:
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate calvin-act
    python eval_env_d.py [--model env_b|env_abc|both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import config
from train_utils import (
    build_policy_features,
    build_rename_map,
    get_device,
    make_delta_timestamps,
    rename_batch_keys,
    set_seed,
)

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.utils.constants import ACTION


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_dir: Path,
    dataset_metadata: LeRobotDatasetMetadata,
    device: torch.device,
) -> tuple[ACTPolicy, callable, callable]:
    """Load a trained ACT policy and its pre/post processors."""
    for ckpt_name in ["checkpoint_best", "checkpoint_final"]:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            print(f"  Loading checkpoint: {ckpt_path}")
            policy = ACTPolicy.from_pretrained(ckpt_path)
            rename_map = build_rename_map(dataset_metadata)
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                dataset_stats=dataset_metadata.stats,
                preprocessor_overrides={
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )
            policy.to(device)
            policy.eval()
            return policy, preprocessor, postprocessor

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    policy: ACTPolicy,
    preprocessor,
    dataloader: DataLoader,
    rename_map: dict[str, str],
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """
    Evaluate an ACT policy on a dataset.

    Uses `predict_action_chunk` to get raw action predictions, then computes
    L1 loss and per-timestep error for chunking robustness analysis.
    """
    policy.eval()

    all_l1: list[float] = []
    per_timestep_l1: list[list[float]] = []

    num_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Rename keys (dataset → policy convention)
            batch = rename_batch_keys(batch, rename_map)
            # Preprocess (normalize, device placement)
            batch = preprocessor(batch)

            # Get predicted action chunk
            actions_hat = policy.predict_action_chunk(batch)  # (B, chunk, dim)
            actions_gt = batch[ACTION]                        # (B, chunk, dim)

            # Align chunk sizes if they differ (e.g., model predicts 30 but Env D has 100)
            min_chunk = min(actions_hat.shape[1], actions_gt.shape[1])
            if actions_hat.shape[1] != actions_gt.shape[1]:
                actions_hat = actions_hat[:, :min_chunk, :]
                actions_gt = actions_gt[:, :min_chunk, :]
            # Per-sample L1
            l1_per_sample = torch.abs(actions_hat - actions_gt).mean(dim=[-2, -1])
            all_l1.extend(l1_per_sample.cpu().tolist())

            # Per-timestep L1 (across the chunk)
            chunk_size = actions_hat.shape[1]
            if len(per_timestep_l1) == 0:
                per_timestep_l1 = [[] for _ in range(chunk_size)]

            for t in range(chunk_size):
                err = torch.abs(actions_hat[:, t, :] - actions_gt[:, t, :]).mean().item()
                per_timestep_l1[t].append(err)

            num_processed += actions_hat.shape[0]

    # ---- Aggregate ----
    mean_l1 = float(np.mean(all_l1)) if all_l1 else float("nan")
    std_l1 = float(np.std(all_l1)) if all_l1 else float("nan")
    mse = float(np.mean(np.array(all_l1) ** 2)) if all_l1 else float("nan")

    per_t_mean = [float(np.mean(lst)) if lst else 0.0 for lst in per_timestep_l1]
    per_t_std = [float(np.std(lst)) if lst else 0.0 for lst in per_timestep_l1]

    half = len(per_t_mean) // 2
    early_mean = float(np.mean(per_t_mean[:half])) if half > 0 else 0.0
    late_mean = float(np.mean(per_t_mean[half:])) if half > 0 else 0.0
    growth = late_mean - early_mean

    return {
        "num_samples": num_processed,
        "mean_l1_loss": mean_l1,
        "std_l1_loss": std_l1,
        "mean_mse_loss": mse,
        "per_timestep_l1_mean": per_t_mean,
        "per_timestep_l1_std": per_t_std,
        "early_chunk_mean_l1": early_mean,
        "late_chunk_mean_l1": late_mean,
        "chunk_error_growth": growth,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation on Environment D"
    )
    parser.add_argument(
        "--model", choices=["env_b", "env_abc", "both"], default="both",
    )
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(config.TRAINING_CONFIG["seed"])
    device = get_device()
    print(f"[eval] Using device: {device}")

    # ---- Init wandb ----
    wb_cfg = config.WANDB_CONFIG
    wandb.login(key=wb_cfg["api_key"], relogin=True, verify=True)
    wandb.init(
        project=wb_cfg["project"],
        entity=wb_cfg.get("entity"),
        name="zero-shot-eval-env-d",
        mode=wb_cfg.get("mode", "online"),
        tags=wb_cfg.get("tags", []) + ["eval", "zero-shot"],
        notes="Zero-shot evaluation on unseen Environment D",
        dir=str(config.OUTPUT_ROOT),
    )

    # ---- Load Environment D metadata ----
    print(f"\n[eval] Loading Environment D: {config.DATASET_ENV_D}")
    d_meta = LeRobotDatasetMetadata(config.DATASET_ENV_D)
    print(f"[eval]   Episodes: {d_meta.total_episodes}, Frames: {d_meta.total_frames}")

    # ---- Build features & rename map for Env D ----
    rename_map = build_rename_map(d_meta)

    # Delta timestamps for dataset loading (dataset-native keys)
    from lerobot.policies.act import ACTConfig
    input_f, output_f = build_policy_features(d_meta)
    tmp_cfg = ACTConfig(input_features=input_f, output_features=output_f)

    delta_ts = {
        "actions": make_delta_timestamps(tmp_cfg.action_delta_indices, d_meta.fps),
    }
    for key, ft in d_meta.features.items():
        if ft.get("dtype", "") in ("image", "video"):
            delta_ts[key] = make_delta_timestamps(
                tmp_cfg.observation_delta_indices, d_meta.fps
            )

    d_dataset = LeRobotDataset(
        config.DATASET_ENV_D,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )
    d_loader = DataLoader(
        d_dataset,
        batch_size=config.EVAL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=config.EVAL_CONFIG["num_workers"],
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    results = {}

    # ---- Evaluate Env-B model ----
    if args.model in ("env_b", "both"):
        print("\n" + "=" * 70)
        print("Evaluating: Env-B-Only Model on Environment D (Zero-Shot)")
        print("=" * 70)

        policy_b, pre_b, post_b = load_model_from_checkpoint(
            config.OUTPUT_ENV_B, d_meta, device
        )
        result_b = evaluate_model(
            policy_b, pre_b, d_loader, rename_map, device, args.max_batches
        )
        results["env_b"] = result_b

        print(f"\n  --- Env-B Model Results on Env D ---")
        print(f"  Samples:          {result_b['num_samples']}")
        print(f"  Mean L1:          {result_b['mean_l1_loss']:.6f} ± {result_b['std_l1_loss']:.6f}")
        print(f"  Mean MSE:         {result_b['mean_mse_loss']:.6f}")
        print(f"  Early-chunk L1:   {result_b['early_chunk_mean_l1']:.6f}")
        print(f"  Late-chunk L1:    {result_b['late_chunk_mean_l1']:.6f}")
        print(f"  Chunk err growth: {result_b['chunk_error_growth']:.6f}")

        out_path = config.OUTPUT_ENV_B / "eval_env_d_results.json"
        with open(out_path, "w") as f:
            json.dump(result_b, f, indent=2)
        print(f"  Saved: {out_path}")

    # ---- Evaluate Env-ABC model ----
    if args.model in ("env_abc", "both"):
        print("\n" + "=" * 70)
        print("Evaluating: Env-ABC Joint Model on Environment D (Zero-Shot)")
        print("=" * 70)

        policy_abc, pre_abc, post_abc = load_model_from_checkpoint(
            config.OUTPUT_ENV_ABC, d_meta, device
        )
        result_abc = evaluate_model(
            policy_abc, pre_abc, d_loader, rename_map, device, args.max_batches
        )
        results["env_abc"] = result_abc

        print(f"\n  --- Env-ABC Model Results on Env D ---")
        print(f"  Samples:          {result_abc['num_samples']}")
        print(f"  Mean L1:          {result_abc['mean_l1_loss']:.6f} ± {result_abc['std_l1_loss']:.6f}")
        print(f"  Mean MSE:         {result_abc['mean_mse_loss']:.6f}")
        print(f"  Early-chunk L1:   {result_abc['early_chunk_mean_l1']:.6f}")
        print(f"  Late-chunk L1:    {result_abc['late_chunk_mean_l1']:.6f}")
        print(f"  Chunk err growth: {result_abc['chunk_error_growth']:.6f}")

        out_path = config.OUTPUT_ENV_ABC / "eval_env_d_results.json"
        with open(out_path, "w") as f:
            json.dump(result_abc, f, indent=2)
        print(f"  Saved: {out_path}")

    # ---- Comparative summary ----
    if len(results) == 2:
        r_b = results["env_b"]
        r_abc = results["env_abc"]
        l1_diff = r_abc["mean_l1_loss"] - r_b["mean_l1_loss"]
        l1_rel = (l1_diff / r_b["mean_l1_loss"]) * 100 if r_b["mean_l1_loss"] else 0

        # ---- Log comparison metrics to wandb ----
        wandb.run.summary["env_b_mean_l1"] = r_b["mean_l1_loss"]
        wandb.run.summary["env_abc_mean_l1"] = r_abc["mean_l1_loss"]
        wandb.run.summary["l1_relative_change_pct"] = l1_rel
        wandb.run.summary["env_b_chunk_growth"] = r_b["chunk_error_growth"]
        wandb.run.summary["env_abc_chunk_growth"] = r_abc["chunk_error_growth"]
        wandb.run.summary["env_b_early_l1"] = r_b["early_chunk_mean_l1"]
        wandb.run.summary["env_abc_early_l1"] = r_abc["early_chunk_mean_l1"]
        wandb.run.summary["env_b_late_l1"] = r_b["late_chunk_mean_l1"]
        wandb.run.summary["env_abc_late_l1"] = r_abc["late_chunk_mean_l1"]

        # Log bar chart: comparison metrics
        metric_names = ["Mean L1", "Mean MSE", "Early-chunk L1", "Late-chunk L1", "Chunk Error Growth"]
        wandb.log({
            "eval/comparison_bar": wandb.plot.bar(
                wandb.Table(
                    columns=["metric", "model", "value"],
                    data=[
                        [metric_names[0], "Env-B Only", r_b["mean_l1_loss"]],
                        [metric_names[0], "Env-ABC Joint", r_abc["mean_l1_loss"]],
                        [metric_names[1], "Env-B Only", r_b["mean_mse_loss"]],
                        [metric_names[1], "Env-ABC Joint", r_abc["mean_mse_loss"]],
                        [metric_names[2], "Env-B Only", r_b["early_chunk_mean_l1"]],
                        [metric_names[2], "Env-ABC Joint", r_abc["early_chunk_mean_l1"]],
                        [metric_names[3], "Env-B Only", r_b["late_chunk_mean_l1"]],
                        [metric_names[3], "Env-ABC Joint", r_abc["late_chunk_mean_l1"]],
                        [metric_names[4], "Env-B Only", r_b["chunk_error_growth"]],
                        [metric_names[4], "Env-ABC Joint", r_abc["chunk_error_growth"]],
                    ],
                ),
                label="metric",
                value="value",
                title="Zero-Shot Evaluation: Env-B vs Env-ABC on Environment D",
            )
        })

        # Log per-timestep error as a line plot
        per_t_data = []
        for t, (err_b, err_abc) in enumerate(zip(
            r_b.get("per_timestep_l1_mean", []),
            r_abc.get("per_timestep_l1_mean", []),
        )):
            per_t_data.append([t, "Env-B Only", err_b])
            per_t_data.append([t, "Env-ABC Joint", err_abc])

        wandb.log({
            "eval/per_timestep_error": wandb.plot.line(
                wandb.Table(
                    columns=["timestep", "model", "l1_error"],
                    data=per_t_data,
                ),
                x="timestep",
                y="l1_error",
                title="Per-Timestep Action L1 Error Across Predicted Chunk (Env D, Zero-Shot)",
            )
        })

        print("\n" + "=" * 70)
        print("Comparative Summary: Env-B vs Env-ABC on Environment D")
        print("=" * 70)
        print(f"  {'Metric':<28} {'Env-B Only':<16} {'Env-ABC Joint':<16} {'Δ':<12}")
        print(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*12}")
        print(f"  {'Mean L1':<28} {r_b['mean_l1_loss']:<16.6f} {r_abc['mean_l1_loss']:<16.6f} {l1_diff:+.6f}")
        print(f"  {'Mean MSE':<28} {r_b['mean_mse_loss']:<16.6f} {r_abc['mean_mse_loss']:<16.6f} {r_abc['mean_mse_loss']-r_b['mean_mse_loss']:+.6f}")
        print(f"  {'Early-chunk L1':<28} {r_b['early_chunk_mean_l1']:<16.6f} {r_abc['early_chunk_mean_l1']:<16.6f} {r_abc['early_chunk_mean_l1']-r_b['early_chunk_mean_l1']:+.6f}")
        print(f"  {'Late-chunk L1':<28} {r_b['late_chunk_mean_l1']:<16.6f} {r_abc['late_chunk_mean_l1']:<16.6f} {r_abc['late_chunk_mean_l1']-r_b['late_chunk_mean_l1']:+.6f}")
        print(f"  {'Chunk Err Growth':<28} {r_b['chunk_error_growth']:<16.6f} {r_abc['chunk_error_growth']:<16.6f} {r_abc['chunk_error_growth']-r_b['chunk_error_growth']:+.6f}")
        print(f"\n  Relative L1 improvement: {l1_rel:+.2f}%")
        print(f"  (Negative = ABC model better)")

        combined_path = config.OUTPUT_ROOT / "eval_env_d_combined.json"
        combined = {
            "dataset": str(config.DATASET_ENV_D),
            "env_b_results": r_b,
            "env_abc_results": r_abc,
            "comparison": {
                "l1_diff": l1_diff,
                "l1_relative_change_pct": l1_rel,
                "mse_diff": r_abc["mean_mse_loss"] - r_b["mean_mse_loss"],
                "chunk_growth_diff": r_abc["chunk_error_growth"] - r_b["chunk_error_growth"],
            },
        }
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Combined results: {combined_path}")

    wandb.finish()

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
