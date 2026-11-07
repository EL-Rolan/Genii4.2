"""Convenience entry point for the DOE command-line application."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

COMMANDS = {
    "train": "train.py",
    "infer": "infer.py",
    "classical": "classical_multi_plane_doe.py",
    "check-images": "dataset.py",
    "mnist-info": "mnist_dataset.py",
    "check-install": "check_install.py",
}

USAGE = """Multi-plane phase-DOE project

Usage:
  python main.py <command> [arguments for that command]

Commands:
  train          Train the PyTorch model
  infer          Predict a DOE from pic1.png ... pic4.png
  classical      Run the classical Gerchberg-Saxton baseline
  check-images   Validate the four input PNG files
  mnist-info     Download and inspect the MNIST dataset
  check-install  Check Fresnel parity and PyTorch autograd

Examples:
  python main.py check-install
  python main.py mnist-info
  python main.py train --epochs 10 --batch-size 4
  python main.py infer --input-dir . --checkpoint checkpoints/best.pt
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(USAGE)
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        print(USAGE)
        raise SystemExit(2)

    script = Path(__file__).resolve().parent / COMMANDS[command]
    subprocess.run([sys.executable, str(script), *sys.argv[2:]], check=True)


if __name__ == "__main__":
    main()
