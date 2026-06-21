#!/usr/bin/env python3
"""
Task 2: Multi-Environment Joint Training on Environments A+B+C.

Trains an ACT policy using combined data from CALVIN Environments A, B, and C
(splitABC_merged dataset). Uses the same architecture and hyperparameters as the
Environment-B-only model for fair comparison.

Usage:
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate calvin-act
    python train_env_abc.py
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
    print("Task 2: Training ACT Policy on Environments A+B+C (Joint)")
    print("=" * 70)
    print(f"Dataset:      {config.DATASET_ENV_ABC}")
    print(f"Output:       {config.OUTPUT_ENV_ABC}")
    print(f"Chunk size:   {config.ACT_CONFIG['chunk_size']}")
    print(f"Epochs:       {config.TRAINING_CONFIG['num_epochs']}")
    print(f"Batch size:   {config.TRAINING_CONFIG['batch_size']}")
    print(f"Learning rate: {config.TRAINING_CONFIG['learning_rate']}")
    print("=" * 70)

    policy, logger = train_act_policy(
        dataset_path=config.DATASET_ENV_ABC,
        output_dir=config.OUTPUT_ENV_ABC,
        run_name="env-abc-joint",
        config_module=config,
    )

    print()
    print("=" * 70)
    print("Task 2 Complete: Model trained on Environments A+B+C")
    print(f"Model saved to: {config.OUTPUT_ENV_ABC}")
    print("=" * 70)


if __name__ == "__main__":
    main()
