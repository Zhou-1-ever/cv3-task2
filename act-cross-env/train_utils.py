"""
Shared training utilities for ACT policy training on CALVIN dataset.

Key design decision:
  The CALVIN v3.0 dataset uses flat keys (e.g., "state", "image", "actions")
  but LeRobot v0.5.2 policies expect "observation."-prefixed keys for observations
  and "action" (singular) for actions.  We rename keys after loading from the
  dataset and before feeding them into the preprocessor / policy.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Key-renaming bridge between v3.0 dataset and v0.5.2 policy convention
# ---------------------------------------------------------------------------

def build_rename_map(dataset_metadata: LeRobotDatasetMetadata) -> dict[str, str]:
    """
    Build a mapping from v3.0 flat dataset keys to v0.5.2 policy keys.

    Dataset keys (v3.0)          Policy keys (v0.5.2)
    ─────────────────────────    ──────────────────────────
    "state"               →      "observation.state"
    "image"               →      "observation.images.image"
    "wrist_image"         →      "observation.images.wrist_image"
    "actions"             →      "action"
    """
    rename = {}
    for key, ft in dataset_metadata.features.items():
        dtype = ft.get("dtype", "")
        if key == "state":
            rename[key] = OBS_STATE  # "observation.state"
        elif dtype in ("image", "video"):
            # "image" → "observation.images.image"
            rename[key] = f"{OBS_IMAGES}.{key}"
        elif key == "actions":
            rename[key] = ACTION  # "action"
    return rename


def rename_batch_keys(batch: dict, rename_map: dict[str, str]) -> dict:
    """Rename keys in a batch according to the rename_map.

    Also handles ``_is_pad`` suffixed keys (e.g. ``actions_is_pad`` →
    ``action_is_pad``) by applying the same mapping to the base key.
    """
    renamed = {}
    for key, value in batch.items():
        if key in rename_map:
            renamed[rename_map[key]] = value
        elif key.endswith("_is_pad"):
            base = key[: -len("_is_pad")]
            if base in rename_map:
                renamed[rename_map[base] + "_is_pad"] = value
            else:
                renamed[key] = value
        else:
            renamed[key] = value
    return renamed


# ---------------------------------------------------------------------------
# Build policy features from v3.0 dataset metadata
# ---------------------------------------------------------------------------

def build_policy_features(
    dataset_metadata: LeRobotDatasetMetadata,
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """
    Build input_features and output_features dicts for ACTConfig.

    Uses policy-convention keys so the ACT model can find action/state/image
    features by their standard names.
    """
    input_features: dict[str, PolicyFeature] = {}
    output_features: dict[str, PolicyFeature] = {}

    for key, ft in dataset_metadata.features.items():
        dtype = ft.get("dtype", "")
        shape = tuple(ft["shape"])

        if dtype in ("image", "video"):
            # Convert (H, W, C) → (C, H, W)
            names = ft.get("names", [])
            if len(shape) == 3 and len(names) >= 3 and names[2] in ("channel", "channels"):
                shape = (shape[2], shape[0], shape[1])
            # Policy key: "observation.images.<camera_name>"
            policy_key = f"{OBS_IMAGES}.{key}"
            input_features[policy_key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)

        elif key == "state":
            input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=shape)

        elif key == "actions":
            output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    return input_features, output_features


# ---------------------------------------------------------------------------
# Training state tracking
# ---------------------------------------------------------------------------

class TrainingLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "training_metrics.jsonl"
        self.records: list[dict] = []

    def log(self, **kwargs) -> None:
        record = {"timestamp": time.time(), **kwargs}
        self.records.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self, extra: dict | None = None) -> str:
        summary_path = self.output_dir / "training_summary.json"
        summary = {"num_records": len(self.records), "records": self.records}
        if extra:
            summary.update(extra)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return str(summary_path)


# ---------------------------------------------------------------------------
# Video frame cache — dramatically speeds up training after the first epoch
# ---------------------------------------------------------------------------
_video_frame_cache: dict[tuple, torch.Tensor] = {}
_video_cache_hits: int = 0
_video_cache_misses: int = 0


def _enable_video_cache() -> None:
    """Monkey-patch the video decoder to cache decoded frames in memory.

    The first epoch decodes from video normally and populates the cache.
    Subsequent epochs (with shuffle) access the same frames in different
    order, yielding near-100% cache hit rate.
    """
    from lerobot.datasets import video_utils

    _original_decode = video_utils.decode_video_frames

    def _cached_decode(
        video_path, timestamps, tolerance_s, backend, return_uint8=True
    ):
        global _video_cache_hits, _video_cache_misses
        key = (str(video_path), tuple(round(t, 3) for t in timestamps))
        if key in _video_frame_cache:
            _video_cache_hits += 1
            return _video_frame_cache[key]
        _video_cache_misses += 1
        result = _original_decode(video_path, timestamps, tolerance_s, backend, return_uint8)
        _video_frame_cache[key] = result
        return result

    video_utils.decode_video_frames = _cached_decode
    # Also patch the module-level reference used by dataset_reader
    import lerobot.datasets.dataset_reader as dr
    dr.decode_video_frames = _cached_decode


def _log_cache_stats() -> None:
    total = _video_cache_hits + _video_cache_misses
    if total > 0:
        hit_rate = 100 * _video_cache_hits / total
        print(f"[cache] Video frame cache: {_video_cache_hits} hits / {total} total = {hit_rate:.1f}% hit rate")
        print(f"[cache] Cached {len(_video_frame_cache)} unique frames")


# ---------------------------------------------------------------------------
# Checkpoint resume helpers
# ---------------------------------------------------------------------------

def _save_resume_state(
    output_dir: Path,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    epoch: int,
    step: int,
    best_loss: float,
) -> None:
    """Save model + state so training can resume after interruption."""
    ckpt_dir = output_dir / "checkpoint_resume"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    state = {"epoch": epoch, "step": step, "best_loss": best_loss}
    with open(output_dir / "training_state.json", "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Weights & Biases helpers
# ---------------------------------------------------------------------------

def _init_wandb(
    run_name: str,
    config_dict: dict,
    output_dir: Path,
    config_module=None,
) -> None:
    """Initialize a wandb run with the given configuration."""
    if config_module is None:
        import config as _cfg
        config_module = _cfg

    wb_cfg = config_module.WANDB_CONFIG
    wandb.login(key=wb_cfg["api_key"], relogin=True, verify=True)
    wandb.init(
        project=wb_cfg["project"],
        entity=wb_cfg.get("entity"),
        name=run_name,
        mode=wb_cfg.get("mode", "online"),
        tags=wb_cfg.get("tags", []),
        notes=wb_cfg.get("notes", ""),
        dir=str(output_dir),
        config=config_dict,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_act_policy(
    dataset_path: Path,
    output_dir: Path,
    run_name: str = "act-training",
    act_config_overrides: dict | None = None,
    training_config_overrides: dict | None = None,
    config_module=None,
) -> tuple[ACTPolicy, TrainingLogger]:
    if config_module is None:
        import config as _cfg
        config_module = _cfg

    act_kwargs = dict(config_module.ACT_CONFIG)
    if act_config_overrides:
        act_kwargs.update(act_config_overrides)

    train_kwargs = dict(config_module.TRAINING_CONFIG)
    if training_config_overrides:
        train_kwargs.update(training_config_overrides)

    set_seed(train_kwargs["seed"])
    device = get_device()
    print(f"[train] Using device: {device}")

    # ---- Enable video frame cache (huge speedup after epoch 1) ----
    _enable_video_cache()

    # ---- Init wandb ----
    _init_wandb(
        run_name=run_name,
        config_dict={
            "dataset": str(dataset_path),
            **act_kwargs,
            **train_kwargs,
        },
        output_dir=output_dir,
        config_module=config_module,
    )

    # ---- Load dataset metadata ----
    print(f"[train] Loading dataset from: {dataset_path}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_path)
    print(f"[train]   Episodes: {dataset_metadata.total_episodes}")
    print(f"[train]   Frames:   {dataset_metadata.total_frames}")
    print(f"[train]   FPS:      {dataset_metadata.fps}")

    # ---- Build policy features (with policy-convention keys) ----
    input_features, output_features = build_policy_features(dataset_metadata)
    print(f"[train] Input features:  {list(input_features.keys())}")
    print(f"[train] Output features: {list(output_features.keys())}")

    # ---- Build key-rename map (dataset → policy keys) ----
    rename_map = build_rename_map(dataset_metadata)
    print(f"[train] Rename map: {rename_map}")

    # ---- Create ACT config ----
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        **{k: v for k, v in act_kwargs.items()
           if k in ACTConfig.__dataclass_fields__},
    )
    if "normalization_mapping" in act_kwargs:
        cfg.normalization_mapping = {
            k: NormalizationMode[v]
            for k, v in act_kwargs["normalization_mapping"].items()
        }

    # ---- Policy, pre/post processors ----
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[train] Total params:    {total_params:,}")
    print(f"[train] Trainable params: {trainable_params:,}")

    # Build preprocessor with rename_map override so it can map
    # "state" → "observation.state" etc.
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    # ---- Delta timestamps (use dataset-native keys for LeRobotDataset) ----
    delta_timestamps = {
        "actions": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # NOTE: cfg.image_features returns policy keys like
    # "observation.images.image", but the dataset uses "image".  We iterate
    # the raw metadata to get the dataset-native image keys.
    for key, ft in dataset_metadata.features.items():
        if ft.get("dtype", "") in ("image", "video"):
            delta_timestamps[key] = make_delta_timestamps(
                cfg.observation_delta_indices, dataset_metadata.fps
            )

    # ---- Dataset & DataLoader (pyav backend because torchcodec lacks FFmpeg) ----
    dataset = LeRobotDataset(
        dataset_path,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    # ---- Apply stride to reduce dataset size ----
    stride = train_kwargs.get("dataset_stride", 1)
    if stride > 1:
        indices = list(range(0, len(dataset), stride))
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"[train] Dataset strided: {len(indices)} samples (stride={stride}, was {len(dataset) // stride * stride})")
        print(f"[train] Batches per epoch: ~{len(indices) // train_kwargs['batch_size']}")

    # Use file_system sharing to avoid /dev/shm 64MB limit
    multiprocessing.set_sharing_strategy("file_system")
    dataloader = DataLoader(
        dataset,
        batch_size=train_kwargs["batch_size"],
        shuffle=True,
        num_workers=train_kwargs["num_workers"],
        pin_memory=device.type != "cpu",
        prefetch_factor=1,
        persistent_workers=True,
        drop_last=True,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ---- Resume from checkpoint if available ----
    state_file = output_dir / "training_state.json"
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    resume_ckpt = output_dir / "checkpoint_resume"
    if resume_ckpt.exists() and state_file.exists():
        print("[train] Found checkpoint — resuming from previous run...")
        policy = ACTPolicy.from_pretrained(resume_ckpt)
        policy.train()
        policy.to(device)
        # Reload pre/post processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        # Rebuild optimizer with reloaded policy params
        optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
        # Load state
        with open(state_file) as f:
            saved = json.load(f)
        start_epoch = saved.get("epoch", 0) + 1  # resume from next epoch
        global_step = saved.get("step", 0)
        best_loss = saved.get("best_loss", float("inf"))
        print(f"[train] Resuming from epoch {start_epoch}, step {global_step}, best_loss={best_loss:.4f}")

    # ---- Training state ----
    logger = TrainingLogger(output_dir)
    num_epochs = train_kwargs["num_epochs"]
    log_interval = train_kwargs["log_interval"]
    save_interval = train_kwargs["save_interval"]

    print(f"[train] Starting training: epoch {start_epoch} → {num_epochs}")
    print(f"[train] Steps per epoch: ~{len(dataloader)}")
    print(f"[train] Total steps: ~{len(dataloader) * num_epochs}")

    training_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_l1_losses = []
        epoch_kl_losses = []

        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", unit="batch", leave=False)
        for batch in epoch_pbar:
            # ---- Key rename: dataset flat keys → policy convention ----
            batch = rename_batch_keys(batch, rename_map)

            # ---- Preprocess (normalize, move to device, etc.) ----
            batch = preprocessor(batch)

            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            l1_loss = loss_dict.get("l1_loss", 0.0)
            kld_loss = loss_dict.get("kld_loss", 0.0)
            if hasattr(l1_loss, "item"):
                l1_loss = l1_loss.item()
            if hasattr(kld_loss, "item"):
                kld_loss = kld_loss.item()
            total_loss = loss.item()

            epoch_pbar.set_postfix({"loss": f"{total_loss:.3f}", "l1": f"{l1_loss:.3f}"})

            epoch_losses.append(total_loss)
            epoch_l1_losses.append(l1_loss)
            epoch_kl_losses.append(kld_loss)

            if global_step % log_interval == 0:
                elapsed = time.time() - training_start
                logger.log(
                    step=global_step, epoch=epoch,
                    loss=total_loss, l1_loss=l1_loss, kld_loss=kld_loss,
                    elapsed=elapsed,
                )
                wandb.log({
                    "train/loss": total_loss,
                    "train/l1_loss": l1_loss,
                    "train/kld_loss": kld_loss,
                    "train/epoch": epoch,
                    "train/elapsed": elapsed,
                }, step=global_step)
                print(
                    f"  [step {global_step:6d} | epoch {epoch:3d}] "
                    f"loss={total_loss:.6f}  l1={l1_loss:.6f}  kld={kld_loss:.6f}"
                )

            if global_step > 0 and global_step % save_interval == 0:
                ckpt_dir = output_dir / f"checkpoint_step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                # Save resume state
                _save_resume_state(output_dir, policy, preprocessor, postprocessor,
                                   epoch, global_step, best_loss)
                print(f"  [checkpoint] Saved to {ckpt_dir}")

            if total_loss < best_loss:
                best_loss = total_loss
                best_dir = output_dir / "checkpoint_best"
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(best_dir)
                preprocessor.save_pretrained(best_dir)
                postprocessor.save_pretrained(best_dir)

            global_step += 1

        avg_loss = np.mean(epoch_losses)
        avg_l1 = np.mean(epoch_l1_losses)
        avg_kl = np.mean(epoch_kl_losses)
        # Save resume state after each epoch
        _save_resume_state(output_dir, policy, preprocessor, postprocessor,
                           epoch, global_step, best_loss)
        print(
            f"--- Epoch {epoch:3d} complete --- "
            f"avg_loss={avg_loss:.6f}  avg_l1={avg_l1:.6f}  avg_kld={avg_kl:.6f}"
        )

    elapsed_total = time.time() - training_start
    print(f"[train] Training complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")

    # Log cache stats
    _log_cache_stats()

    # Log final summary to wandb
    wandb.run.summary["best_loss"] = best_loss
    wandb.run.summary["total_steps"] = global_step
    wandb.run.summary["total_time_s"] = elapsed_total
    wandb.finish()

    final_dir = output_dir / "checkpoint_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    print(f"[train] Final model saved to {final_dir}")

    logger.save_summary(extra={
        "total_steps": global_step,
        "best_loss": best_loss,
        "total_time_s": elapsed_total,
    })

    return policy, logger
