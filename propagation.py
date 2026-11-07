import math
from typing import Sequence

import torch
from torch import nn

WAVELENGTH_M = 532e-9
DOE_PIXEL_SIZE_M = 10e-6
DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
#БПФ исп. в train и infer
def centered_fft2(x: torch.Tensor) -> torch.Tensor:
    #Прямое БПФ с центрированием спектра
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )

class FresnelPropagator(nn.Module):
    def __init__(self, wavelength_m: float = WAVELENGTH_M, doe_pixel_size_m: float = DOE_PIXEL_SIZE_M, distances_m: Sequence[float] = DISTANCES_M) -> None:
        super().__init__()
        if wavelength_m <= 0 or doe_pixel_size_m <= 0:
            raise ValueError("Длина волны и размер пикселя ДОЭ должны быть положительными")
        if len(distances_m) != 4 or any(float(z) <= 0 for z in distances_m):
            raise ValueError("Требуются ровно четыре положительных расстояния распространения")
        self.wavelength_m = float(wavelength_m)
        self.doe_pixel_size_m = float(doe_pixel_size_m)
        self.distances_m = tuple(float(z) for z in distances_m)

    def forward(self, phase: torch.Tensor, return_fields: bool = False):
        if phase.ndim == 4 and phase.shape[1] == 1:
            phase = phase[:, 0]
        if phase.ndim != 3 or phase.shape[-1] != phase.shape[-2]:
            raise ValueError("Фаза должна иметь форму [B,N,N] или [B,1,N,N]")

        batch, n, _ = phase.shape
        real_dtype = phase.dtype if phase.dtype in (torch.float32, torch.float64) else torch.float32
        phase = phase.to(dtype=real_dtype)
        device = phase.device
        k = 2.0 * math.pi / self.wavelength_m
        dx_in = self.doe_pixel_size_m

        coords_in = (torch.arange(n, device=device, dtype=real_dtype) - n // 2) * dx_in  #Входная сетка с центром в нуле
        y_in, x_in = torch.meshgrid(coords_in, coords_in, indexing="ij")
        field_in = torch.polar(torch.ones_like(phase), phase)
        amplitudes = []
        fields = []

        for z in self.distances_m:
            dx_out = self.wavelength_m * z / (n * dx_in)
            coords_out = (torch.arange(n, device=device, dtype=real_dtype) - n // 2) * dx_out  #Выходная сетка с центром в нуле
            y_out, x_out = torch.meshgrid(coords_out, coords_out, indexing="ij")
            chirp_in = torch.exp(1j * (k * (x_in.square() + y_in.square()) / (2.0 * z)))
            chirp_out = torch.exp(1j * (k * (x_out.square() + y_out.square()) / (2.0 * z)))
            prefactor = complex(math.cos(k * z), math.sin(k * z)) / (1j * self.wavelength_m * z)

            spectrum = centered_fft2(field_in * chirp_in)
            field_out = prefactor * chirp_out * spectrum * (dx_in ** 2)
            fields.append(field_out)
            amplitudes.append(torch.abs(field_out))

        stacked_amplitudes = torch.stack(amplitudes, dim=1)
        if return_fields:
            return stacked_amplitudes, torch.stack(fields, dim=1)
        return stacked_amplitudes

def normalize_plane_shapes(amplitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if amplitudes.ndim != 4:
        raise ValueError("Ожидается амплитуда формы [B,4,H,W]")
    norms = torch.linalg.vector_norm(amplitudes.flatten(2), dim=2).clamp_min(eps)
    return amplitudes / norms[:, :, None, None]
