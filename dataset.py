"""программа ищет рядом с dataset.py файлы: pic1.png, pic2.png, pic3.png, pic4.png
Формат каждого файла должен соответствовать кейсу: PNG, 128x128 пикселей, grayscale."""

import argparse
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image

# Параметры задачи. Порядок расстояний соответствует pic1.png ... pic4.png.
DEFAULT_Z_POSITIONS_MM: tuple[float, ...] = (24.1, 25.3, 26.5, 27.7)
DEFAULT_FILENAMES: tuple[str, ...] = (
    "pic1.png",
    "pic2.png",
    "pic3.png",
    "pic4.png",
)
DEFAULT_IMAGE_SIZE: tuple[int, int] = (128, 128)


class HologramDataset: #Исп.класс, чтобы получить один объект для посл.работы с DataLoader
    def __init__(
        self,
        data_dir: str | Path = Path(__file__).resolve().parent,
        filenames: Sequence[str] = DEFAULT_FILENAMES,
        z_positions_mm: Sequence[float] = DEFAULT_Z_POSITIONS_MM,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        png_representation: Literal["intensity", "amplitude"] = "intensity",
        resize: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.filenames = tuple(filenames)
        self.z_positions_mm = tuple(float(z) for z in z_positions_mm)
        self.image_size = tuple(int(v) for v in image_size)
        self.png_representation = png_representation
        self.resize = resize

        if len(self.filenames) != len(self.z_positions_mm):
            raise ValueError(
                "Количество файлов должно совпадать с количеством расстояний z."
            )
        if not self.filenames:
            raise ValueError("Список входных изображений пуст.")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size должен иметь вид (width, height) из положительных чисел.")
        if png_representation not in ("intensity", "amplitude"):
            raise ValueError("png_representation должен быть 'intensity' или 'amplitude'.")

        self.paths = [self.data_dir / name for name in self.filenames]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            missing_text = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Не найдены входные изображения:\n"
                f"{missing_text}\n"
                "Положите pic1.png ... pic4.png рядом с dataset.py "
                "или укажите папку через --data-dir."
            )

        self.images: list[np.ndarray] = [self._load_image(path) for path in self.paths]
        self.amplitudes: list[np.ndarray] = [
            self._make_amplitude(image) for image in self.images
        ]

    def _load_image(self, path: Path) -> np.ndarray:
        try:
            with Image.open(path) as source:
                image = source.convert("L") #Конвертация в L даёт один 8-битный канал яркости
        except Exception as exc:
            raise ValueError(f"Не удалось открыть изображение {path}: {exc}") from exc

        if image.size != self.image_size:
            if not self.resize:
                raise ValueError(
                    f"Неверный размер файла {path.name}: {image.size}; "
                    f"ожидается {self.image_size}. "
                    "Исправьте файл или создайте dataset с resize=True."
                )
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.shape != (self.image_size[1], self.image_size[0]):
            raise RuntimeError(
                f"Внутренняя ошибка размера {path.name}: получено {array.shape}."
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def _make_amplitude(self, image: np.ndarray) -> np.ndarray: #Преобразует яркость PNG в амплитуду целевого поля
        if self.png_representation == "intensity":
            amplitude = np.sqrt(np.clip(image, 0.0, 1.0))
        else:
            amplitude = np.clip(image, 0.0, 1.0)
        return np.ascontiguousarray(amplitude, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        return {
            "image": self.images[index].copy(),
            "target_amplitude": self.amplitudes[index].copy(),
            "z_mm": np.float32(self.z_positions_mm[index]),
            "index": np.int64(index),
            "path": str(self.paths[index]),
        }

    def images_as_array(self) -> np.ndarray: #Возвращает все изображения одним массивом формы (4, 128, 128)
        return np.stack(self.images, axis=0).astype(np.float32, copy=False)

    def amplitudes_as_array(self) -> np.ndarray:
        return np.stack(self.amplitudes, axis=0).astype(np.float32, copy=False)

def export_mnist_examples(
    output_dir: str | Path = Path(__file__).resolve().parent,
    indices: Sequence[int] = (0, 1, 2, 3),
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mnist_dir: str | Path | None = None,
) -> list[Path]:   #Сохраняет выбранные изображения MNIST как pic1.png....pic4.png
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError(
            "Для загрузки MNIST установите PyTorch и torchvision: "
            "pip install torch torchvision"
        ) from exc

    if len(indices) != 4:
        raise ValueError("Для данного проекта нужно выбрать ровно 4 изображения MNIST.")
    if any(int(i) < 0 for i in indices):
        raise ValueError("Индексы MNIST должны быть неотрицательными.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(mnist_dir) if mnist_dir else output_dir / "mnist_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mnist = MNIST(root=str(cache_dir), train=True, download=True)
    saved_paths: list[Path] = []

    for number, dataset_index in enumerate(indices, start=1):
        if int(dataset_index) >= len(mnist):
            raise IndexError(
                f"Индекс MNIST {dataset_index} выходит за диапазон 0..{len(mnist) - 1}."
            )

        pil_image, label = mnist[int(dataset_index)]
        pil_image = pil_image.convert("L").resize(
            image_size, Image.Resampling.BILINEAR
        )
        output_path = output_dir / f"pic{number}.png"
        pil_image.save(output_path, format="PNG")
        saved_paths.append(output_path)
        print(f"Сохранено: {output_path.name} (цифра {label}, индекс MNIST {dataset_index})")
    return saved_paths

def print_dataset_info(dataset: HologramDataset) -> None: #отладочная печать
    print("\nНабор данных загружен успешно")
    print(f"Папка: {dataset.data_dir}")
    print(f"Количество изображений: {len(dataset)}")
    print(f"Размер каждого изображения: {dataset.image_size[0]}x{dataset.image_size[1]}")
    print(f"Представление PNG: {dataset.png_representation}")

    for item in range(len(dataset)):
        sample = dataset[item]
        image = sample["image"]
        amplitude = sample["target_amplitude"]
        assert isinstance(image, np.ndarray)
        assert isinstance(amplitude, np.ndarray)
        print(
            f"{Path(str(sample['path'])).name}: "
            f"z={float(sample['z_mm']):.1f} мм, "
            f"shape={image.shape}, dtype={image.dtype}, "
            f"диапазон яркости=[{image.min():.3f}, {image.max():.3f}], "
            f"диапазон амплитуды=[{amplitude.min():.3f}, {amplitude.max():.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка и проверка четырёх изображений для проекта ДОЭ."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Папка с pic1.png ... pic4.png (по умолчанию рядом с dataset.py).",
    )
    parser.add_argument(
        "--make-mnist",
        action="store_true",
        help="Загрузить MNIST и сохранить четыре выбранных примера в PNG.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs=4,
        default=(0, 1, 2, 3),
        metavar=("I1", "I2", "I3", "I4"),
        help="Четыре индекса примеров в MNIST (по умолчанию: 0 1 2 3).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Разрешить изменение размера входных картинок до 128x128.",
    )
    parser.add_argument(
        "--png-representation",
        choices=("intensity", "amplitude"),
        default="intensity",
        help=(
            "Что кодирует яркость PNG: intensity — интенсивность (амплитуда=sqrt(I)); "
            "amplitude — амплитуда поля. По умолчанию intensity."
        ),
    )
    args = parser.parse_args()

    if args.make_mnist:
        export_mnist_examples(output_dir=args.data_dir, indices=args.indices)

    dataset = HologramDataset(
        data_dir=args.data_dir,
        resize=args.resize,
        png_representation=args.png_representation,
    )
    print_dataset_info(dataset)