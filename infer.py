import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

from model import PhaseDOEUNet
from propagation import FresnelPropagator


IMAGE_SIZE = (128, 128)
FILENAMES = ("pic1.png", "pic2.png", "pic3.png", "pic4.png")
DEFAULT_DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


def load_image(path: Path, representation: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Входное изображение не найдено: {path.resolve()}")

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0

    if representation == "intensity":
        array = np.sqrt(np.clip(array, 0.0, 1.0))
    elif representation != "amplitude":
        raise ValueError(
            f"Неизвестное представление {representation!r}. "
            "Допустимы 'amplitude' и 'intensity'."
        )

    return np.ascontiguousarray(array, dtype=np.float32)


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    phase = np.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)
    phase = np.clip(phase, -np.pi, np.pi)
    return np.rint((phase + np.pi) * (255.0 / (2.0 * np.pi))).astype(np.uint8)


def save_normalized_image(array: np.ndarray, path: Path) -> None:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)

    peak = float(array.max()) if array.size else 0.0
    if peak > 0.0:
        array = array / peak

    image_u8 = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        if torch.cuda.is_available():
            device_argument = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device_argument = "mps"
        else:
            device_argument = "cpu"

    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Запрошен CUDA, но PyTorch не видит CUDA. "
            "Проверь установку CUDA-версии PyTorch; либо запусти с --device cpu."
        )
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Запрошен MPS, но он недоступен в этой установке PyTorch.")
    return device


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Файл контрольной точки не найден: {path.resolve()}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Контрольная точка должна содержать словарь с параметрами модели.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("В checkpoint отсутствует ключ 'model_state_dict'.")
    return checkpoint


def normalize_for_metrics(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(array.max()) if array.size else 0.0
    if maximum > 0.0:
        array = array / maximum
    return np.clip(array, 0.0, 1.0)


def calculate_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.ravel()
    second_flat = second.ravel()
    if np.std(first_flat) < 1e-12 or np.std(second_flat) < 1e-12:
        return float("nan")
    return float(pearsonr(first_flat, second_flat)[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Восстановление четырёх плоскостей с помощью обученной модели ДОЭ"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Папка с файлами pic1.png, pic2.png, pic3.png и pic4.png",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./checkpoints/best.pt"),
        help="Путь к обученной модели",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./neural_results"),
        help="Папка для сохранения ДОЭ, реконструкций и метрик",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Устройство: auto, cuda, cpu или mps",
    )
    parser.add_argument(
        "--representation",
        choices=("checkpoint", "amplitude", "intensity"),
        default="checkpoint",
        help="Интерпретация входных изображений; checkpoint использует настройку обучения",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    representation = (checkpoint.get("representation", "amplitude") if args.representation == "checkpoint" else args.representation)
    if representation not in ("amplitude", "intensity"):
        raise ValueError(
            f"В checkpoint указано неизвестное representation={representation!r}."
        )

    target_arrays = [load_image(args.input_dir / filename, representation) for filename in FILENAMES]
    target = torch.from_numpy(np.stack(target_arrays, axis=0)).unsqueeze(0).to(device)

    base_channels = int(checkpoint.get("base_channels", 32))
    model = PhaseDOEUNet(in_channels=4, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wavelength_m = float(checkpoint.get("wavelength_m", 532e-9))
    doe_pixel_size_m = float(checkpoint.get("doe_pixel_size_m", 10e-6))
    distances_m: Iterable[float] = checkpoint.get("distances_m", DEFAULT_DISTANCES_M)
    distances_m = tuple(float(distance) for distance in distances_m)

    if len(distances_m) != len(FILENAMES):
        raise ValueError(
            f"В checkpoint должно быть {len(FILENAMES)} расстояния, "
            f"но найдено {len(distances_m)}."
        )
    if wavelength_m <= 0.0 or doe_pixel_size_m <= 0.0 or any(z <= 0.0 for z in distances_m):
        raise ValueError("Длина волны, шаг пикселя ДОЭ и расстояния должны быть положительными.")

    propagator = FresnelPropagator(wavelength_m=wavelength_m, doe_pixel_size_m=doe_pixel_size_m, distances_m=distances_m).to(device)

    with torch.inference_mode():
        phase_batch = model(target)

    if phase_batch.ndim == 4 and phase_batch.shape[1] == 1:
        phase_batch = phase_batch[:, 0]
    if phase_batch.ndim != 3 or phase_batch.shape[0] != 1:
        raise RuntimeError(
            "Модель должна вернуть фазу формы [1, 128, 128] "
            "или [1, 1, 128, 128]; получено "
            f"{tuple(phase_batch.shape)}."
        )

    phase = phase_batch[0].detach().cpu().numpy().astype(np.float32)
    if phase.shape != IMAGE_SIZE:
        raise RuntimeError(
            f"Модель вернула фазу размера {phase.shape}, ожидался {IMAGE_SIZE}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / "doe_phase_radians.npy", phase)
    doe_u8 = phase_to_uint8(phase)
    Image.fromarray(doe_u8).save(args.output_dir / "doe.png")

    decoded_phase = doe_u8.astype(np.float32) * (2.0 * math.pi / 255.0) - math.pi
    phase_tensor = torch.from_numpy(decoded_phase).unsqueeze(0).to(device)

    with torch.inference_mode():
        reconstructed_batch = propagator(phase_tensor)

    if reconstructed_batch.ndim == 4:
        if reconstructed_batch.shape[0] != 1:
            raise RuntimeError(
                "Ожидался один пример на входе пропагатора, получена форма "
                f"{tuple(reconstructed_batch.shape)}."
            )
        reconstructed_batch = reconstructed_batch[0]
    if reconstructed_batch.ndim != 3:
        raise RuntimeError(
            "Пропагатор должен вернуть массив формы [4, H, W] "
            "или [1, 4, H, W]; получено "
            f"{tuple(reconstructed_batch.shape)}."
        )

    reconstructed = reconstructed_batch.detach().cpu().numpy()
    if np.iscomplexobj(reconstructed):
        reconstructed = np.abs(reconstructed)

    if reconstructed.shape[0] != len(FILENAMES):
        raise RuntimeError(
            f"Пропагатор вернул {reconstructed.shape[0]} плоскостей вместо "
            f"{len(FILENAMES)}. Полная форма: {reconstructed.shape}."
        )

    rows = []
    for index, (distance_m, target_image, reconstructed_image) in enumerate(
        zip(distances_m, target_arrays, reconstructed), start=1
    ):
        rec_path = args.output_dir / f"reconstructed_{index}.png"
        save_normalized_image(reconstructed_image, rec_path)

        target_norm = normalize_for_metrics(target_image)
        rec_norm = normalize_for_metrics(reconstructed_image)
        if target_norm.shape != rec_norm.shape:
            raise RuntimeError(
                f"Размер целевого изображения {target_norm.shape} не совпадает "
                f"с размером реконструкции {rec_norm.shape} в плоскости {index}."
            )

        pearson = calculate_pearson(target_norm, rec_norm)
        ssim = float(structural_similarity(target_norm, rec_norm, data_range=1.0))

        distance_mm = distance_m * 1e3
        output_pixel_pitch_um = (
            wavelength_m * distance_m / (IMAGE_SIZE[0] * doe_pixel_size_m) * 1e6
        )
        rows.append(
            {
                "plane": index,
                "distance_mm": distance_mm,
                "output_pixel_pitch_um": output_pixel_pitch_um,
                "pearson": pearson,
                "ssim": ssim,
            }
        )
        print(
            f"Плоскость {index}: z={distance_mm:.3f} мм, "
            f"dx_out={output_pixel_pitch_um:.3f} мкм, "
            f"КК={pearson:.5f}, ИСС={ssim:.5f}"
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nГотово.")
    print(f"Устройство: {device}")
    print(f"Представление входных изображений: {representation}")
    print(f"Результаты сохранены в: {args.output_dir.resolve()}")
    print(f"Файл ДОЭ: {(args.output_dir / 'doe.png').resolve()}")
    print(f"Таблица метрик: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
