# Синтез фазовой ДОЭ для четырёх плоскостей

Программа получает **четыре изображения 128×128**, совместно обрабатывает их нейросетью и формирует **одну фазовую ДОЭ 128×128**. Затем рассчитывает восстановленные изображения на четырёх расстояниях по формуле Френеля.

Проект можно запустить двумя способами:

- **В Docker** — зависимости устанавливаются внутри контейнера
- **Локально в Windows** — Python и библиотеки устанавливаются непосредственно на компьютер

Оба способа используют одни и те же скрипты, входные изображения и файл весов `checkpoints/best.pt`.

## Параметры задачи

| Параметр | Значение |
|---|---:|
| Входные изображения | 4 × 128×128, grayscale |
| Выходная ДОЭ | 128×128, PNG, 8 бит, grayscale |
| Длина волны | 532 нм |
| Шаг пикселя ДОЭ | 10 мкм |
| Расстояния восстановления | 24,1; 25,3; 26,5; 27,7 мм |

Файлы должны называться `pic1.png`, `pic2.png`, `pic3.png`, `pic4.png`. Они соответствуют расстояниям в указанном порядке.

## Назначение файлов

- `main.py` — запускает нужный скрипт по команде: обучение, инференс, проверка изображений или классический алгоритм
- `model.py` — архитектура нейросети `PhaseDOEUNet` на основе U-Net; принимает четыре изображения и предсказывает фазу ДОЭ
- `train.py` — обучает нейросеть и сохраняет веса
- `mnist_dataset.py` — формирует обучающие наборы из четырёх изображений MNIST
- `dataset.py` — загружает и проверяет четыре входных изображения; при необходимости может подготовить примеры MNIST
- `infer.py` — загружает веса, создаёт ДОЭ и рассчитывает четыре восстановленных изображения
- `propagation.py` — рассчитывает распространение Френеля в PyTorch
- `GS.py` — классический многоплоскостной алгоритм Гершберга—Сакстона для сравнения с нейросетью
- `Dockerfile` и `requirements.txt` — сборка контейнера и список Python-зависимостей

## Порядок запуска в Windows

Все команды ниже выполняются в **PowerShell из корневой папки проекта**.

### 1. Получить проект

**Docker:** клонируйте репозиторий и перейдите в папку проекта:

```powershell
git clone https://github.com/EL-Rolan/Genii4.2.git
cd Genii4.2
```

**Локально:** скачайте ZIP-архив репозитория с GitHub, распакуйте его и откройте PowerShell в распакованной папке `Genii4.2`. Убедитесь, что рядом находятся `main.py`, `requirements.txt` и остальные файлы проекта. Команда `git clone` для этого способа не нужна.

### 2. Подготовить среду запуска

**Docker:** установите Docker Desktop и запустите его. Проверьте, что команды Docker доступны:

```powershell
docker --version
docker info
```

**Локально:** установите Python 3.11. Затем создайте виртуальное окружение и установите зависимости проекта:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Далее локальные команды запускаются через `.\.venv\Scripts\python.exe`. Активация окружения не требуется.

### 3. Проверить обученные веса

Для инференса обоими способами нужен файл `checkpoints\best.pt`.

```powershell
Test-Path .\checkpoints\best.pt
```

Команда должна вернуть `True`. Если весов нет, обучите модель локально — это можно сделать и на компьютере, который в дальнейшем будет запускать инференс через Docker.

**Локальная команда обучения** (после установки Python-зависимостей на шаге 2):

```powershell
.\.venv\Scripts\python.exe .\main.py train --epochs 10 --batch-size 4 --output-dir .\checkpoints
```

`mnist_dataset.py` автоматически загрузит MNIST при первом запуске обучения. По завершении в `checkpoints` появятся `best.pt`, `last.pt` и `history.csv`. Для инференса нужен `best.pt`.

**Для Docker:** файл `checkpoints\best.pt` должен уже существовать до сборки образа. Dockerfile проверяет его наличие и не соберёт образ без весов.

### 4. Подготовить входные изображения

Для каждого набора создайте отдельную папку. Например:

```text
test_data\
├── set_01\
│   ├── pic1.png
│   ├── pic2.png
│   ├── pic3.png
│   └── pic4.png
└── set_02\
    ├── pic1.png
    ├── pic2.png
    ├── pic3.png
    └── pic4.png
```

Каждое изображение должно иметь размер 128×128 пикселей. Программа преобразует цветные изображения в grayscale. Если размер отличается, добавьте параметр `--resize` к команде инференса.

Проверить один набор можно локальной командой:

```powershell
.\.venv\Scripts\python.exe .\main.py check-images --data-dir .\test_data\set_01
```

### 5. Запустить нейросеть на одном наборе

#### Вариант A — Docker

Соберите образ (это делается один раз или после изменения кода/зависимостей):

```powershell
docker build -t genii-doe .
```

Запустите обработку набора:

```powershell
$inputPath = (Resolve-Path .\test_data\set_01).Path
$outputPath = (New-Item -ItemType Directory -Force .\output\set_01).FullName

docker run --rm `
  --mount "type=bind,source=$inputPath,target=/data/input,readonly" `
  --mount "type=bind,source=$outputPath,target=/data/output" `
  genii-doe
```

#### Вариант B — локальный запуск Python

Docker не нужен. Выполните инференс напрямую:

```powershell
.\.venv\Scripts\python.exe .\main.py infer `
  --input-dir .\test_data\set_01 `
  --output-dir .\output\set_01 `
  --checkpoint .\checkpoints\best.pt
```

### 6. Запустить обработку пяти и более наборов

Положите каждый набор в отдельную папку внутри `test_data` (`set_01`, `set_02` и так далее). В каждой папке должны находиться свои `pic1.png`–`pic4.png`.

#### Вариант A — Docker

```powershell
Get-ChildItem .\test_data -Directory | ForEach-Object {
    $inputPath = $_.FullName
    $setName = $_.Name
    $outputPath = (New-Item -ItemType Directory -Force ".\output\$setName").FullName

    docker run --rm `
      --mount "type=bind,source=$inputPath,target=/data/input,readonly" `
      --mount "type=bind,source=$outputPath,target=/data/output" `
      genii-doe

    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка обработки набора $setName"
    }
}
```

#### Вариант B — локальный запуск Python

```powershell
Get-ChildItem .\test_data -Directory | ForEach-Object {
    $inputPath = $_.FullName
    $setName = $_.Name
    $outputPath = (New-Item -ItemType Directory -Force ".\output\$setName").FullName

    & .\.venv\Scripts\python.exe .\main.py infer `
      --input-dir $inputPath `
      --output-dir $outputPath `
      --checkpoint .\checkpoints\best.pt

    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка обработки набора $setName"
    }
}
```

Результаты каждого набора сохраняются отдельно: `output\set_01`, `output\set_02` и т. д.

## Пример результатов работы программы

После обработки одного набора в `output\set_01` появятся:

```text
output\set_01\
├── doe.png                    # итоговая фазовая ДОЭ
├── doe_phase_radians.npy      # фаза в радианах до квантования
├── reconstructed_1.png       # восстановление на 24,1 мм
├── reconstructed_2.png       # восстановление на 25,3 мм
├── reconstructed_3.png       # восстановление на 26,5 мм
├── reconstructed_4.png       # восстановление на 27,7 мм
└── metrics.csv                # КК (Pearson) и ИСС (SSIM)
```

`doe.png` — выходная ДОЭ в формате PNG, 128×128, 8 бит, grayscale. Файлы `reconstructed_1.png`–`reconstructed_4.png` показывают результат распространения на соответствующих расстояниях.

В `metrics.csv` записываются отдельные значения для четырёх плоскостей. Пример структуры файла:

```csv
plane,distance_mm,output_pixel_pitch_um,pearson,ssim
1,24.1,...,...,...
2,25.3,...,...,...
3,26.5,...,...,...
4,27.7,...,...,...
```

Среднее значение КК и ИСС для набора рассчитывается как среднее четырёх значений соответствующего столбца. Автоматическая сводная таблица по всем наборам не создаётся.

## Решение через алгоритм Гершберга - Сакстона для сравнения

`GS.py` реализует классический многоплоскостной алгоритм Гершберга—Сакстона (ГС) без нейросети. Он начинает со случайной фазовой карты и выполняет 400 итераций. На каждой итерации поле распространяется на четыре плоскости; в каждой плоскости амплитуда заменяется амплитудой соответствующего входного изображения при сохранении текущей фазы. Затем поля распространяются обратно к ДОЭ, объединяются, и фазовая карта обновляется по их общей фазе. Так алгоритм постепенно подбирает одну фазовую ДОЭ, согласованную с четырьмя плоскостями.

Для запуска ГС подготовьте четыре файла `pic1.png`–`pic4.png` **в корневой папке проекта рядом с `GS.py`**, затем выполните локально:

```powershell
.\.venv\Scripts\python.exe .\main.py classical
```

Результаты классического алгоритма сохраняются в папке `results`. Этот запуск использует локальный Python и установленные зависимости; Docker для него не требуется.

## Как рассчитывается распространение Френеля

В `propagation.py` и `GS.py` для центрированной сетки применяется прямое двумерное БПФ: перед `fft2` выполняется `ifftshift`, после — `fftshift`. Координатные сетки отсчитываются от центра массива. Выходной шаг пикселя определяется выражением:

```text
dx_out = wavelength * z / (N * dx_in)
```

где `N` — размер стороны ДОЭ, `dx_in` — шаг пикселя ДОЭ, `z` — расстояние распространения.

## Прим.

- Использован датасет рукописных цифр MNIST
- используется cuda pytorch
- Параметры обученной модели: Checkpoint: E:\NuclearITHack\checkpoints\best.pt
Epoch: 10
Loss: None
Representation: amplitude
Base channels: 32
Wavelength: 5.32e-07
Pixel size: 1e-05
Distances: (0.0241, 0.0253, 0.0265, 0.0277)
