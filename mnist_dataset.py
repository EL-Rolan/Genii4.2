import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_SIZE = 128
DEFAULT_Z_MM = (24.1, 25.3, 26.5, 27.7)

class FourPlaneMNIST(Dataset):
    def __init__(
        self,
        root: str | Path = "./data",
        train: bool = True,
        download: bool = True,
        representation: str = "amplitude",
        seed: int = 1234,
    ) -> None:
        if representation not in {"amplitude", "intensity"}:
            raise ValueError("representation должен быть amplitude или intensity")
        self.representation = representation
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
        ])
        self.mnist = datasets.MNIST(
            root=str(root), train=train, download=download, transform=transform
        )
        generator = torch.Generator().manual_seed(seed + (0 if train else 1))
        self.indices = torch.randperm(len(self.mnist), generator=generator).tolist()  #Перемешивание с фиксир. seed
        self.indices = self.indices[: (len(self.indices) // 4) * 4]
        self.z_mm = torch.tensor(DEFAULT_Z_MM, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices) // 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        images, labels = [], []
        for plane in range(4):
            mnist_index = self.indices[index * 4 + plane]
            image, label = self.mnist[mnist_index]  #Форма изображения: [1, 128, 128]
            image = image.squeeze(0)
            if self.representation == "intensity":
                image = torch.sqrt(image.clamp_min(0.0))
            images.append(image)
            labels.append(label)
        return {
            "target_amplitude": torch.stack(images, dim=0),  #Форма группы: [4, H, W]
            "labels": torch.tensor(labels, dtype=torch.long),
            "z_mm": self.z_mm.clone(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка и просмотр MNIST для проекта ДОЭ")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--representation", choices=("amplitude", "intensity"), default="amplitude")
    args = parser.parse_args()

    train_set = FourPlaneMNIST(args.data_dir, train=True, representation=args.representation)
    test_set = FourPlaneMNIST(args.data_dir, train=False, representation=args.representation)
    sample = train_set[0]
    print(f"MNIST загружен и сохранён в: {Path(args.data_dir).resolve()}")
    print(f"Групп из четырёх изображений для обучения: {len(train_set)}")
    print(f"Групп из четырёх изображений для проверки: {len(test_set)}")
    print(f"Размер тензора примера: {tuple(sample['target_amplitude'].shape)}")
    print(f"Метки: {sample['labels'].tolist()}")
    print(f"Расстояния z (мм): {sample['z_mm'].tolist()}")
    print(f"Диапазон значений: [{sample['target_amplitude'].min().item():.3f}, {sample['target_amplitude'].max().item():.3f}]")


if __name__ == "__main__":
    main()