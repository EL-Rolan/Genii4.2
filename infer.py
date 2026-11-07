import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

from model import PhaseDOEUNet
from propagation import FresnelPropagator

FILENAMES = ("pic1.png", "pic2.png", "pic3.png", "pic4.png")

def load_image(path: Path, resize: bool, representation: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Входное изображение не найдено: {path}")
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (128, 128):
            if not resize:
                raise ValueError(f"Файл {path.name} имеет размер {im.size}; ожидается (128, 128). Для изменения размера используйте --resize.")
            im = im.resize((128, 128), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    if representation == "intensity":
        arr = np.sqrt(np.clip(arr, 0.0, 1.0))
    return np.ascontiguousarray(arr)


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    phase = np.clip(phase, -np.pi, np.pi)
    return np.rint((phase + np.pi) * (255.0 / (2.0 * np.pi))).astype(np.uint8)


def save_normalized_image(array: np.ndarray, path: Path) -> None:
    array = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    peak = float(array.max()) if array.size else 0.0
    if peak > 0:
        array = array / peak
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск обученной модели ДОЭ на PyTorch")
    parser.add_argument("--input-dir", default=".", help="Папка с файлами pic1.png ... pic4.png")
    parser.add_argument("--checkpoint", default="./checkpoints/best.pt")
    parser.add_argument("--output-dir", default="./neural_results")
    parser.add_argument("--resize", action="store_true", help="Изменять размер изображений, отличных от 128x128")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--representation", choices=("checkpoint", "amplitude", "intensity"), default="checkpoint")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    representation = checkpoint.get("representation", "amplitude") if args.representation == "checkpoint" else args.representation
    input_dir = Path(args.input_dir)
    target_arrays = [load_image(input_dir / name, args.resize, representation) for name in FILENAMES]
    target = torch.from_numpy(np.stack(target_arrays)).unsqueeze(0).to(device)

    model = PhaseDOEUNet(in_channels=4, base_channels=int(checkpoint.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    propagator = FresnelPropagator(
        wavelength_m=float(checkpoint.get("wavelength_m", 532e-9)),
        doe_pixel_size_m=float(checkpoint.get("doe_pixel_size_m", 10e-6)),
        distances_m=checkpoint.get("distances_m", (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)),
    ).to(device)

    with torch.no_grad():
        phase = model(target)[0].cpu().numpy()  #Одна фазовая карта для четырёх входных изображений

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "doe_phase_radians.npy", phase.astype(np.float32))
    doe_u8 = phase_to_uint8(phase)
    Image.fromarray(doe_u8, mode="L").save(output_dir / "doe.png")  #PNG, 8 бит, оттенки серого

    decoded_phase = doe_u8.astype(np.float32) * (2.0 * math.pi / 255.0) - math.pi
    phase_tensor = torch.from_numpy(decoded_phase).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = propagator(phase_tensor)[0].cpu().numpy()

    rows = []
    for index, (z_mm, target_amp, rec_amp) in enumerate(
        zip((24.1, 25.3, 26.5, 27.7), target_arrays, reconstructed), start=1
    ):
        rec_path = output_dir / f"reconstructed_{index}.png"
        save_normalized_image(rec_amp, rec_path)
        rec_norm = rec_amp / max(float(rec_amp.max()), 1e-12)
        target_norm = target_amp / max(float(target_amp.max()), 1e-12)
        pearson = float(pearsonr(target_norm.ravel(), rec_norm.ravel())[0]) if np.std(target_norm) and np.std(rec_norm) else float("nan")
        ssim = float(structural_similarity(target_norm, rec_norm, data_range=1.0))
        dx_out_um = propagator.wavelength_m * (z_mm * 1e-3) / (128 * propagator.doe_pixel_size_m) * 1e6
        rows.append({"plane": index, "distance_mm": z_mm, "output_pixel_pitch_um": dx_out_um, "pearson": pearson, "ssim": ssim})
        print(f"Плоскость {index}: z={z_mm:.1f} мм, dx_out={dx_out_um:.3f} мкм, КК={pearson:.5f}, ИСС={ssim:.5f}")

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"ДОЭ и восстановленные изображения сохранены в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()