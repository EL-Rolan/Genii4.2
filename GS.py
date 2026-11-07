"""Решение алгоритмом Гершберга - Сакстона
400 итераций
"""

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
IMAGE_SIZE = 128
INPUT_FILENAMES = ("pic1.png", "pic2.png", "pic3.png", "pic4.png")

WAVELENGTH_M = 532e-9       #532нм
DOE_PIXEL_SIZE_M = 10e-6    #10мкм
DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)

ITERATIONS = 400
RELAXATION = 0.7            #Коэфф. релаксации
RANDOM_SEED = 7
INPUT_IMAGES_ARE_INTENSITY = False

OUTPUT_DIRNAME = "results"

def load_grayscale_image(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден файл {path.name}. Положите его рядом со скриптом."
        )

    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0

    if array.shape != (size, size):
        raise ValueError(f"{path.name}: ожидался размер {(size, size)}, получен {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"В {path.name} обнаружены NaN или inf.")
    return np.clip(array, 0.0, 1.0)

def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    clipped = np.clip(phase, -np.pi, np.pi)
    encoded = np.rint((clipped + np.pi) * (255.0 / (2.0 * np.pi)))
    return np.clip(encoded, 0, 255).astype(np.uint8)

def uint8_to_phase(encoded: np.ndarray) -> np.ndarray:
    return encoded.astype(np.float64) * (2.0 * np.pi / 255.0) - np.pi

def save_grayscale_png(array_01: np.ndarray, path: Path) -> None:
    array = np.asarray(array_01, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    max_value = float(array.max()) if array.size else 0.0
    if max_value > 0.0:
        array = array / max_value
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path, format="PNG")

def centered_fft2(array: np.ndarray) -> np.ndarray:
    #Прямое БПФ на центрированной сетке
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))

def centered_ifft2(array: np.ndarray) -> np.ndarray:
    #Обратное БПФ на центрированной сетке
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array)))

def coordinate_grid(size: int, pixel_pitch: float) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (np.arange(size, dtype=np.float64) - size // 2) * pixel_pitch
    return np.meshgrid(coordinates, coordinates, indexing="xy")

def fresnel_propagate_one_fft(
    field_at_doe: np.ndarray,
    distance_m: float,
    wavelength_m: float = WAVELENGTH_M,
    doe_pixel_size_m: float = DOE_PIXEL_SIZE_M,
) -> tuple[np.ndarray, float]:
    if field_at_doe.ndim != 2 or field_at_doe.shape[0] != field_at_doe.shape[1]:
        raise ValueError("Поле ДОЭ должно быть квадратной двумерной матрицей.")
    if distance_m <= 0 or wavelength_m <= 0 or doe_pixel_size_m <= 0:
        raise ValueError("Длина волны, расстояние и размер пикселя должны быть положительными.")

    n = field_at_doe.shape[0]
    k = 2.0 * np.pi / wavelength_m
    dx_in = doe_pixel_size_m
    dx_out = wavelength_m * distance_m / (n * dx_in)

    x_in, y_in = coordinate_grid(n, dx_in)
    x_out, y_out = coordinate_grid(n, dx_out)

    chirp_in = np.exp(1j * k * (x_in**2 + y_in**2) / (2.0 * distance_m))
    chirp_out = np.exp(1j * k * (x_out**2 + y_out**2) / (2.0 * distance_m))
    prefactor = np.exp(1j * k * distance_m) / (1j * wavelength_m * distance_m)

    spectrum = centered_fft2(field_at_doe * chirp_in)
    field_out = prefactor * chirp_out * spectrum * (dx_in**2)
    return field_out, dx_out

def fresnel_backpropagate_one_fft(
    field_at_output: np.ndarray,
    distance_m: float,
    wavelength_m: float = WAVELENGTH_M,
    doe_pixel_size_m: float = DOE_PIXEL_SIZE_M,
) -> np.ndarray:
    n = field_at_output.shape[0]
    k = 2.0 * np.pi / wavelength_m
    dx_in = doe_pixel_size_m
    dx_out = wavelength_m * distance_m / (n * dx_in)

    x_in, y_in = coordinate_grid(n, dx_in)
    x_out, y_out = coordinate_grid(n, dx_out)

    chirp_in = np.exp(1j * k * (x_in**2 + y_in**2) / (2.0 * distance_m))
    chirp_out = np.exp(1j * k * (x_out**2 + y_out**2) / (2.0 * distance_m))
    prefactor = np.exp(1j * k * distance_m) / (1j * wavelength_m * distance_m)

    spectrum = field_at_output / (prefactor * chirp_out * (dx_in**2))
    field_before_chirp = centered_ifft2(spectrum)
    return field_before_chirp * np.conj(chirp_in)

def prepare_target_amplitudes(images: Iterable[np.ndarray], distances_m: Iterable[float],) -> tuple[list[np.ndarray], list[float]]:
    targets: list[np.ndarray] = []
    output_pixel_sizes: list[float] = []
    n = IMAGE_SIZE
    input_power = (n * n) * (DOE_PIXEL_SIZE_M**2)

    for image, z in zip(images, distances_m):
        desired_amplitude = np.asarray(image, dtype=np.float64)
        if INPUT_IMAGES_ARE_INTENSITY:
            desired_amplitude = np.sqrt(np.clip(desired_amplitude, 0.0, 1.0))

        dx_out = WAVELENGTH_M * z / (n * DOE_PIXEL_SIZE_M)
        target_power = np.sum(desired_amplitude**2) * (dx_out**2)
        if target_power <= np.finfo(np.float64).eps:
            raise ValueError("Одно из входных изображений полностью чёрное; нельзя задать его амплитуду.")

        scale = math.sqrt(input_power / target_power)
        targets.append(desired_amplitude * scale)
        output_pixel_sizes.append(dx_out)

    return targets, output_pixel_sizes


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def normalized_shape_error(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    actual_norm = np.linalg.norm(actual)
    target_norm = np.linalg.norm(target)
    if actual_norm == 0 or target_norm == 0:
        return float("inf")
    actual = actual / actual_norm
    target = target / target_norm
    return float(np.sum((actual - target) ** 2))


def synthesize_phase_doe( target_images: list[np.ndarray], distances_m: tuple[float, ...] = DISTANCES_M, iterations: int = ITERATIONS, relaxation: float = RELAXATION, seed: int = RANDOM_SEED,) -> np.ndarray:
    if len(target_images) != 4 or len(distances_m) != 4:
        raise ValueError("Алгоритм ожидает ровно четыре изображения и четыре расстояния.")
    if iterations < 1:
        raise ValueError("iterations должно быть >= 1.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation должно находиться в диапазоне (0, 1].")

    target_amplitudes, _ = prepare_target_amplitudes(target_images, distances_m)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=(IMAGE_SIZE, IMAGE_SIZE))

    best_phase = phase.copy()
    best_error = float("inf")

    for iteration in range(1, iterations + 1):
        field_at_doe = np.exp(1j * phase)  #Поле фазового ДОЭ
        backpropagated_fields = []
        errors = []

        for target_amplitude, z in zip(target_amplitudes, distances_m):
            field_at_plane, _ = fresnel_propagate_one_fft(field_at_doe, z)
            current_amplitude = np.abs(field_at_plane)
            errors.append(normalized_shape_error(current_amplitude, target_amplitude))
            constrained_field = target_amplitude * np.exp(1j * np.angle(field_at_plane))
            field_estimate_at_doe = fresnel_backpropagate_one_fft(constrained_field, z)
            backpropagated_fields.append(field_estimate_at_doe)

        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_phase = phase.copy()

        mean_backpropagated = np.mean(backpropagated_fields, axis=0)
        proposed_phase = np.angle(mean_backpropagated)

        phase_difference = np.angle(np.exp(1j * (proposed_phase - phase)))
        phase = wrap_phase(phase + relaxation * phase_difference)

        if iteration == 1 or iteration % 20 == 0 or iteration == iterations:
            per_plane = " | ".join(f"{value:.5g}" for value in errors)
            print(
                f"Итерация {iteration:4d}/{iterations}: "
                f"ошибка по плоскостям [{per_plane}], средняя={mean_error:.6g}"
            )

    print(f"Лучшая средняя ошибка по ходу итераций: {best_error:.6g}")
    return best_phase

def pearson_correlation(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(reconstructed, dtype=np.float64).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1) Чтение четырёх исходных изображений...")
    input_paths = [script_dir / name for name in INPUT_FILENAMES]
    original_images = [load_grayscale_image(path) for path in input_paths]
    print("   Все изображения приведены к 128x128, оттенки серого, float32 [0, 1].")

    print("2) Синтез одной фазовой ДОЭ многоплоскостным методом Гершберга—Сакстона (ГС)...")
    phase = synthesize_phase_doe(original_images)

    np.save(output_dir / "doe_phase_radians.npy", phase.astype(np.float32))

    doe_uint8 = phase_to_uint8(phase)
    doe_path = output_dir / "doe.png"
    Image.fromarray(doe_uint8, mode="L").save(doe_path, format="PNG")
    print(f"   ДОЭ сохранена: {doe_path}")

    with Image.open(doe_path) as doe_image:
        decoded_doe = np.asarray(doe_image.convert("L"), dtype=np.uint8)
    quantized_phase = uint8_to_phase(decoded_doe)
    doe_field = np.exp(1j * quantized_phase)

    print("3) Восстановление четырёх изображений из сохранённой 8-битной ДОЭ...")
    metric_rows = []

    for index, (z, original) in enumerate(zip(DISTANCES_M, original_images), start=1):
        field_at_plane, dx_out = fresnel_propagate_one_fft(doe_field, z)

        reconstructed_amplitude = np.abs(field_at_plane)
        reconstructed_path = output_dir / f"reconstructed_{index}.png"
        save_grayscale_png(reconstructed_amplitude, reconstructed_path)

        with Image.open(reconstructed_path) as reconstructed_image:
            reconstructed_u8 = np.asarray(reconstructed_image.convert("L"), dtype=np.uint8)
        reconstructed_01 = reconstructed_u8.astype(np.float32) / 255.0

        pearson = pearson_correlation(original, reconstructed_01)
        ssim = float(
            structural_similarity(
                original.astype(np.float32),
                reconstructed_01.astype(np.float32),
                data_range=1.0,
            )
        )

        metric_rows.append(
            {
                "image": index,
                "distance_mm": z * 1e3,
                "output_pixel_pitch_um": dx_out * 1e6,
                "pearson": pearson,
                "ssim": ssim,
            }
        )

        print(
            f"   {reconstructed_path.name}: z={z * 1e3:.1f} mm, "
            f"dx_out={dx_out * 1e6:.3f} um, "
            f"КК={pearson:.5f}, ИСС={ssim:.5f}"
        )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "distance_mm", "output_pixel_pitch_um", "pearson", "ssim"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print("\nГотово. Результаты находятся в папке:", output_dir)
    print("  doe.png                  — фазовая ДОЭ, 8 бит, оттенки серого")
    print("  doe_phase_radians.npy    — фаза в радианах (float)")
    print("  reconstructed_1..4.png   — четыре восстановленных изображения |U|")
    print("  metrics.csv              — КК и ИСС для каждой пары")

if __name__ == "__main__":
    main()