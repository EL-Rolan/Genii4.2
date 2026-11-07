"""Train the phase-DOE network using a differentiable Fresnel model."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mnist_dataset import FourPlaneMNIST
from model import PhaseDOEUNet
from propagation import FresnelPropagator, normalize_plane_shapes


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, propagator, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0

    for batch in loader:
        target = batch["target_amplitude"].to(device, non_blocking=True)  # [B,4,H,W]
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            phase = model(target)
            reconstructed = propagator(phase)
            loss = F.mse_loss(
                normalize_plane_shapes(reconstructed),
                normalize_plane_shapes(target),
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        count = target.shape[0]
        total_loss += float(loss.detach().item()) * count
        total_items += count

    return total_loss / max(total_items, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch multi-plane DOE network")
    parser.add_argument("--data-dir", default="./data", help="MNIST cache directory")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="0 = use full MNIST train split; smaller values are useful for a smoke run")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="0 = use full MNIST test split")
    parser.add_argument("--representation", choices=("amplitude", "intensity"), default="amplitude",
                        help="amplitude matches the current classical script; intensity applies sqrt(I)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Image representation: {args.representation}")

    train_dataset = FourPlaneMNIST(args.data_dir, train=True, representation=args.representation, seed=args.seed)
    test_dataset = FourPlaneMNIST(args.data_dir, train=False, representation=args.representation, seed=args.seed)
    if args.max_train_samples > 0:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_test_samples > 0:
        test_dataset = Subset(test_dataset, range(min(args.max_test_samples, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = PhaseDOEUNet(in_channels=4, base_channels=args.base_channels).to(device)
    propagator = FresnelPropagator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "learning_rate"])

    print(f"Training groups: {len(train_dataset)} | Test groups: {len(test_dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, propagator, train_loader, device, optimizer)
        test_loss = run_epoch(model, propagator, test_loader, device)
        scheduler.step(test_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.7f} test={test_loss:.7f} lr={lr:.2e}")

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, test_loss, lr])

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "base_channels": args.base_channels,
            "representation": args.representation,
            "image_size": 128,
            "wavelength_m": propagator.wavelength_m,
            "doe_pixel_size_m": propagator.doe_pixel_size_m,
            "distances_m": propagator.distances_m,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if test_loss < best_val:
            best_val = test_loss
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Training finished. Best test loss: {best_val:.7f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Loss history: {history_path}")


if __name__ == "__main__":
    main()
