import os
import shutil
import random

# Your source folder
SOURCE = r"Z:\DownTown\banana_dataset-20260526T131959Z-3-001\banana_dataset"

# Destination
DEST = r"C:\Users\Piyush Keshav Jha\Desktop\banana_project\dataset"

# Classes
CLASSES = ["ripe", "unripe", "overripe", "rotten"]

# Split ratios
TRAIN = 0.70
VAL   = 0.15
TEST  = 0.15

random.seed(42)

for cls in CLASSES:
    src_folder = os.path.join(SOURCE, cls)
    images = os.listdir(src_folder)
    images = [f for f in images if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)

    total = len(images)
    t = int(total * TRAIN)
    v = int(total * VAL)

    splits = {
        "train": images[:t],
        "val":   images[t:t+v],
        "test":  images[t+v:]
    }

    for split, files in splits.items():
        dest_folder = os.path.join(DEST, split, cls)
        for f in files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(dest_folder, f))
        print(f"✅ {cls} → {split}: {len(files)} images")

print("\n🍌 Dataset split complete!")