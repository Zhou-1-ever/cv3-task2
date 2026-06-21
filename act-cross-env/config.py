"""Shared configuration for ACT policy training on CALVIN dataset."""
from pathlib import Path
SPEED_PRESET: str = "full"
DATA_ROOT = Path("/inspire/hdd/project/fdu-aidake-cfff/public/fangyukai/final_pj/Task2/data/calvin-lerobot")
OUTPUT_ROOT = Path("/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/task2_act/outputs")
DATASET_ENV_B = DATA_ROOT / "splitB"
DATASET_ENV_ABC = DATA_ROOT / "splitABC_merged"
DATASET_ENV_D = DATA_ROOT / "splitD"
OUTPUT_ENV_B = OUTPUT_ROOT / "model_env_b"
OUTPUT_ENV_ABC = OUTPUT_ROOT / "model_env_abc"
_PRESETS = {
    "fast": {"chunk_size": 10, "n_action_steps": 10, "num_epochs": 15, "dataset_stride": 500, "description": "~15 min/model"},
    "medium": {"chunk_size": 15, "n_action_steps": 15, "num_epochs": 20, "dataset_stride": 200, "description": "~1 hour/model"},
    "full": {"chunk_size": 30, "n_action_steps": 30, "num_epochs": 30, "dataset_stride": 50, "description": "overnight"},
}
_p = _PRESETS[SPEED_PRESET]
ACT_CONFIG = {
    "n_obs_steps": 1, "chunk_size": _p["chunk_size"], "n_action_steps": _p["n_action_steps"],
    "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    "vision_backbone": "resnet18", "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "replace_final_stride_with_dilation": False, "pre_norm": False, "dim_model": 512,
    "n_heads": 8, "dim_feedforward": 3200, "feedforward_activation": "relu",
    "n_encoder_layers": 4, "n_decoder_layers": 1, "use_vae": True, "latent_dim": 32,
    "n_vae_encoder_layers": 4, "temporal_ensemble_coeff": None, "dropout": 0.1, "kl_weight": 10.0,
}
TRAINING_CONFIG = {
    "batch_size": 16, "learning_rate": 1e-5, "weight_decay": 1e-4, "lr_backbone": 1e-5,
    "num_epochs": _p["num_epochs"], "log_interval": 10, "save_interval": 500,
    "num_workers": 2, "dataset_stride": _p["dataset_stride"], "seed": 42,
}
EVAL_CONFIG = {"batch_size": 16, "num_workers": 2, "max_eval_batches": None}
WANDB_CONFIG = {
    "api_key": "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",
    "project": "act-cross-env-generalization", "entity": None, "mode": "online",
    "tags": ["ACT", "CALVIN", "cross-env-generalization", SPEED_PRESET],
    "notes": f"ACT cross-env generalization -- preset: {SPEED_PRESET}",
}
