from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from bird_song import DEFAULT_CLASSES
from bird_song.generation.evaluation import detail_metrics
from bird_song.generation.wgan_gp import (
    ConditionalCritic,
    ConditionalGenerator,
    WGANConfig,
    count_parameters,
    critic_loss,
    generator_loss,
    gradient_penalty,
)
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.vocoder import VocoderMelScaler, VocoderSpectrogramConfig
from bird_song.vocoder_data import BigVGANMelDataset, make_mel_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def autocast_context(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


@torch.inference_mode()
def evaluate_epoch(
    generator: ConditionalGenerator,
    critic: ConditionalCritic,
    loader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    generator.eval()
    critic.eval()
    real_scores: list[float] = []
    fake_scores: list[float] = []
    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []
    for batch_index, (real, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fake = generator.sample(labels)
        real_scores.append(float(critic(real, labels).mean()))
        fake_scores.append(float(critic(fake, labels).mean()))
        real_batches.append(real.cpu())
        fake_batches.append(fake.cpu())
    if not real_batches:
        raise RuntimeError("validation loader produced no batches")
    real_images = torch.cat(real_batches)
    fake_images = torch.cat(fake_batches)
    metrics = detail_metrics(real_images, fake_images)
    metrics.update(
        {
            "real_score": sum(real_scores) / len(real_scores),
            "fake_score": sum(fake_scores) / len(fake_scores),
        }
    )
    metrics["wasserstein_gap"] = metrics["real_score"] - metrics["fake_score"]
    metrics["selection_error"] = sum(
        abs(math.log(max(metrics[name], 1e-8)))
        for name in ("time_detail_ratio", "frequency_detail_ratio", "sample_std_ratio")
    )
    metrics["selection_metric"] = "sum_abs_log_detail_and_sample_std_ratios"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a rectangular conditional WGAN-GP on BigVGAN mels.")
    parser.add_argument("--cache-manifest", type=Path, default=PROJECT_ROOT / "artifacts/bigvgan_mels/mel_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/bigvgan_mels")
    parser.add_argument("--vocoder-config", type=Path, default=PROJECT_ROOT / "configs/bigvgan_spectrogram.json")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/wgan_gp.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/wgan_gp_bigvgan")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--critic-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers cannot be negative")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    if args.eval_batches is not None and args.eval_batches < 1:
        raise ValueError("eval-batches must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    cache_manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(DEFAULT_CLASSES)
    if set(cache_manifest["name"].unique()) != set(classes):
        raise ValueError("cache classes do not match the three expected bird species")
    vocoder_config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    model_config = WGANConfig.from_dict(json.loads(args.model_config.read_text(encoding="utf-8")))
    if (model_config.height, model_config.width) != (vocoder_config.n_mels, vocoder_config.expected_frames):
        raise ValueError("WGAN and BigVGAN mel shapes do not match")
    scaler = VocoderMelScaler.from_json(args.cache_root / "scaler.json")
    train_set = BigVGANMelDataset(args.cache_manifest, args.cache_root, "train", classes, vocoder_config, scaler)
    validation_set = BigVGANMelDataset(args.cache_manifest, args.cache_root, "validation", classes, vocoder_config, scaler)
    train_loader = make_mel_loader(train_set, args.batch_size, args.workers, training=True, seed=args.seed)
    validation_loader = make_mel_loader(validation_set, args.batch_size, args.workers)
    generator = ConditionalGenerator(model_config).to(device)
    critic = ConditionalCritic(model_config).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_learning_rate or args.learning_rate, betas=(0.0, 0.9)
    )
    critic_steps = args.critic_steps or model_config.critic_steps
    if critic_steps < 1:
        raise ValueError("critic-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in ("config.json", "history.csv", "last.pt") if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {[path.name for path in existing]}")
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "classes": classes,
            "model": model_config.to_dict(),
            "vocoder": vocoder_config.to_dict(),
            "scaler": scaler.to_dict(),
            "generator_parameters": count_parameters(generator),
            "critic_parameters": count_parameters(critic),
            "critic_steps": critic_steps,
            "device_used": str(device),
        },
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history_path = args.output_dir / "history.csv"
    best_error = float("inf")
    fields = [
        "epoch", "critic_loss", "generator_loss", "gradient_penalty", "validation_gap",
        "time_detail_ratio", "frequency_detail_ratio", "sample_std_ratio", "selection_error", "seconds",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            generator.train()
            critic.train()
            critic_losses: list[float] = []
            generator_losses: list[float] = []
            penalties: list[float] = []
            for batch_index, (real, labels, _) in enumerate(train_loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                real = real.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                for _ in range(critic_steps):
                    critic_optimizer.zero_grad(set_to_none=True)
                    noise = torch.randn(real.shape[0], model_config.latent_dim, device=device)
                    with autocast_context(device):
                        fake = generator(noise, labels).detach()
                        real_score = critic(real, labels)
                        fake_score = critic(fake, labels)
                    with torch.autocast(device_type=device.type, enabled=False):
                        penalty = gradient_penalty(critic, real.float(), fake.float(), labels)
                        loss = critic_loss(real_score.float(), fake_score.float(), penalty, model_config.gradient_penalty_weight)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(critic_optimizer)
                    scaler_amp.update()
                    critic_losses.append(float(loss.detach()))
                    penalties.append(float(penalty.detach()))
                generator_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(real.shape[0], model_config.latent_dim, device=device)
                with autocast_context(device):
                    fake = generator(noise, labels)
                    loss = generator_loss(critic(fake, labels))
                scaler_amp.scale(loss).backward()
                scaler_amp.step(generator_optimizer)
                scaler_amp.update()
                generator_losses.append(float(loss.detach()))
            validation = evaluate_epoch(generator, critic, validation_loader, device, args.eval_batches)
            row = {
                "epoch": epoch,
                "critic_loss": sum(critic_losses) / max(len(critic_losses), 1),
                "generator_loss": sum(generator_losses) / max(len(generator_losses), 1),
                "gradient_penalty": sum(penalties) / max(len(penalties), 1),
                "validation_gap": validation["wasserstein_gap"],
                "time_detail_ratio": validation["time_detail_ratio"],
                "frequency_detail_ratio": validation["frequency_detail_ratio"],
                "sample_std_ratio": validation["sample_std_ratio"],
                "selection_error": validation["selection_error"],
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            history_file.flush()
            print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
            checkpoint = {
                "format_version": 2,
                "model_type": "conditional_wgan_gp_bigvgan_mel_generator",
                "generator_state": generator.state_dict(),
                "critic_state": critic.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "classes": list(classes),
                "vocoder_config": vocoder_config.to_dict(),
                "scaler": scaler.to_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "metrics": row | validation,
            }
            atomic_torch_save(checkpoint, args.output_dir / "last.pt")
            if row["selection_error"] < best_error:
                best_error = row["selection_error"]
                atomic_torch_save(
                    {
                        "format_version": 2,
                        "model_type": checkpoint["model_type"],
                        "generator_state": cpu_state_dict(generator),
                        "model_config": model_config.to_dict(),
                        "classes": list(classes),
                        "vocoder_config": vocoder_config.to_dict(),
                        "scaler": scaler.to_dict(),
                        "epoch": epoch,
                        "seed": args.seed,
                        "metrics": row | validation,
                    },
                    args.output_dir / "best_generator.pt",
                )
    print(f"Best validation selection error: {best_error:.4f}")
    print(f"Generator checkpoint: {args.output_dir / 'best_generator.pt'}")


if __name__ == "__main__":
    main()
