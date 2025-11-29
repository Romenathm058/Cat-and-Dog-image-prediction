import os
import shutil
import random

def split_images(source, train_dir, val_dir, split_ratio=0.8):
    if not os.path.exists(source):
        print(f"Source folder not found: {source}")
        return

    files = [f for f in os.listdir(source)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(files) == 0:
        print(f"No valid images found in: {source}")
        return

    random.shuffle(files)

    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    val_files = files[split_point:]

    # Create folders
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy train files
    for file in train_files:
        src = os.path.join(source, file)
        dest = os.path.join(train_dir, file)
        try:
            shutil.copy(src, dest)
        except Exception:
            pass  # skip corrupted images

    # Copy val files
    for file in val_files:
        src = os.path.join(source, file)
        dest = os.path.join(val_dir, file)
        try:
            shutil.copy(src, dest)
        except Exception:
            pass  # skip corrupted images

    print(f"{os.path.basename(source)} → Train: {len(train_files)}, Val: {len(val_files)}")


# ----------- EDIT THESE PATHS ONLY --------------

CAT_SOURCE = "Cat"
DOG_SOURCE = "Dog"

TRAIN_CAT = "data/train/cats"
VAL_CAT   = "data/val/cats"

TRAIN_DOG = "data/train/dogs"
VAL_DOG   = "data/val/dogs"

# -------------------------------------------------

split_images(CAT_SOURCE, TRAIN_CAT, VAL_CAT)
split_images(DOG_SOURCE, TRAIN_DOG, VAL_DOG)