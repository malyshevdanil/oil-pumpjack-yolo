"""Обучение YOLO на разметке из CVAT и вывод метрик.

Нужны: images_for_cvat/ (фото) и сегментированные фотки/practica_2.zip (экспорт CVAT, YOLO Segmentation).
Запуск: pip install ultralytics && python train_and_metrics.py
"""
import random
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
EPOCHS = {"det": 35, "seg": 30}

# 1. Распаковываем разметку из CVAT
labels_dir = ROOT / "cvat_export"
zipfile.ZipFile(ROOT / "сегментированные фотки" / "practica_2.zip").extractall(labels_dir)
labels = sorted((labels_dir / "labels" / "train").glob("*.txt"))

# 2. Делим на train / val / test (70 / 15 / 15)
random.Random(42).shuffle(labels)
n = len(labels)
splits = {"test": labels[:n * 15 // 100], "val": labels[n * 15 // 100:n * 30 // 100], "train": labels[n * 30 // 100:]}


def polygon_to_box(line):
    """Строка полигона YOLO -> строка bbox YOLO (min/max координат контура)."""
    v = line.split()
    xs, ys = [float(x) for x in v[1::2]], [float(y) for y in v[2::2]]
    return f"{v[0]} {(min(xs) + max(xs)) / 2} {(min(ys) + max(ys)) / 2} {max(xs) - min(xs)} {max(ys) - min(ys)}"


# 3. Собираем два датасета: seg (полигоны) и det (bbox из полигонов)
for task in ("seg", "det"):
    for split, files in splits.items():
        (ROOT / "data" / task / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / task / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lbl in files:
            shutil.copy(ROOT / "images_for_cvat" / f"{lbl.stem}.jpg", ROOT / "data" / task / "images" / split)
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if task == "det":
                lines = [polygon_to_box(l) for l in lines]
            (ROOT / "data" / task / "labels" / split / lbl.name).write_text("\n".join(lines))
    (ROOT / "data" / f"{task}.yaml").write_text(
        f"path: {(ROOT / 'data' / task).as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: pumpjack\n", encoding="utf-8")

# 4. Обучаем и считаем метрики на test
for task, weights in (("det", "yolo11n.pt"), ("seg", "yolo11n-seg.pt")):
    model = YOLO(weights)
    model.train(data=str(ROOT / "data" / f"{task}.yaml"), epochs=EPOCHS[task], imgsz=640, batch=8,
                workers=0, seed=42, project=str(ROOT / "runs"), name=task, exist_ok=True)
    # берём веса последней эпохи: best.pt может оказаться ранней эпохой со случайными рамками
    final = YOLO(ROOT / "runs" / task / "weights" / "last.pt")
    m = final.val(data=str(ROOT / "data" / f"{task}.yaml"), split="test",
                  project=str(ROOT / "runs"), name=f"{task}_test", exist_ok=True)
    print(f"\n=== {task} / test ===")
    print(f"bbox: P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    if task == "seg":
        print(f"mask: P={m.seg.mp:.3f} R={m.seg.mr:.3f} mAP50={m.seg.map50:.3f} mAP50-95={m.seg.map:.3f}")
