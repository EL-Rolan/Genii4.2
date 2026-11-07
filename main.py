import sys
import subprocess
from pathlib import Path

COMMANDS = {  #Команды и соответствующие файлы запуска
    "train": "train.py",
    "infer": "infer.py",
    "classical": "GS.py",
    "check-images": "dataset.py",
    "mnist-info": "mnist_dataset.py",
}

USAGE = """Проект синтеза многоплоскостной фазовой ДОЭ

Использование:
  python main.py <команда> [аргументы команды]

Команды:
  train          Обучение модели PyTorch
  infer          Синтез ДОЭ по pic1.png ... pic4.png
  classical      Запуск классического алгоритма Гершберга—Сакстона
  check-images   Проверка четырёх входных PNG
  mnist-info     Загрузка и просмотр набора MNIST

Примеры:
  python main.py mnist-info
  python main.py train --epochs 10 --batch-size 4
  python main.py infer --input-dir . --checkpoint checkpoints/best.pt
  python main.py classical
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(USAGE)
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Неизвестная команда: {command}\n")
        print(USAGE)
        raise SystemExit(2)

    script = Path(__file__).resolve().parent / COMMANDS[command]
    subprocess.run([sys.executable, str(script), *sys.argv[2:]], check=True)


if __name__ == "__main__":
    main()