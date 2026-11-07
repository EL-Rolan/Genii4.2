"""Offline smoke test: model, Fresnel propagation and autograd integration."""
from __future__ import annotations

import numpy as np
import torch

from classical_multi_plane_doe import fresnel_propagate_one_fft
from model import PhaseDOEUNet
from propagation import FresnelPropagator, normalize_plane_shapes


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(7)
    phase_np = rng.uniform(-np.pi, np.pi, (128, 128)).astype(np.float64)
    field_np = np.exp(1j * phase_np)
    expected, _ = fresnel_propagate_one_fft(field_np, 24.1e-3)

    propagator = FresnelPropagator().to(device)
    phase_t = torch.from_numpy(phase_np.astype(np.float32)).unsqueeze(0).to(device)
    actual = propagator(phase_t)[0, 0].detach().cpu().numpy()
    max_diff = float(np.max(np.abs(actual - np.abs(expected))))
    print(f"Fresnel parity max |difference|: {max_diff:.3e}")
    if max_diff > 1e-3:
        raise RuntimeError("PyTorch Fresnel layer does not match the NumPy baseline")

    model = PhaseDOEUNet(base_channels=8).to(device)
    target = torch.rand(2, 4, 128, 128, device=device)
    phase = model(target)
    reconstructed = propagator(phase)
    loss = torch.nn.functional.mse_loss(
        normalize_plane_shapes(reconstructed), normalize_plane_shapes(target)
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Training loss is not finite")
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("A model parameter has a non-finite gradient")

    print(f"Model phase shape: {tuple(phase.shape)}")
    print(f"Reconstruction shape: {tuple(reconstructed.shape)}")
    print(f"Smoke loss: {loss.detach().item():.6g}")
    print("PASS: forward propagation and backward gradients are operational.")


if __name__ == "__main__":
    main()
