#!/usr/bin/env python3
"""
Task 1: Basic ACT Policy Training on Environment B only.

Trains an Action Chunking Transformer (ACT) policy using only CALVIN Environment B
data. This serves as the baseline for cross-environment generalization experiments.

Usage:
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate calvin-act
    python train_env_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project directory is in the path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import config
from train_utils import train_act_policy


def main():
    print("=" * 70)
    print("Task 1: Training ACT Policy on Environment B")
    print("=" * 70)
    print(f"Dataset:      {config.DATASET_ENV_B}")
    print(f"Output:       {config.OUTPUT_ENV_B}")
    print(f"Chunk size:   {config.ACT_CONFIG['chunk_size']}")
    print(f"Epochs:       {config.TRAINING_CONFIG['num_epochs']}")
    print(f"Batch size:   {config.TRAINING_CONFIG['batch_size']}")
    print(f"Learning rate: {config.TRAINING_CONFIG['learning_rate']}")
    print("=" * 70)

    policy, logger = train_act_policy(
        dataset_path=config.DATASET_ENV_B,
        output_dir=config.OUTPUT_ENV_B,
        run_name="env-b-only",
        config_module=config,
    )

    print()
    print("=" * 70)
    print("Task 1 Complete: Model trained on Environment B")
    print(f"Model saved to: {config.OUTPUT_ENV_B}")
    print("=" * 70)


if __name__ == "__main__":
    main()
