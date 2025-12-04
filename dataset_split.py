import os
import shutil
import random

# Source and target directories
SOURCE = "airborne_dataset"
TARGET = "dataset"

# 80/20 train-val split (common default for initial experiments)
TRAIN_SPLIT = 0.8

# Create main split folders if they don't exist yet
os.makedirs(os.path.join(TARGET, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "val"), exist_ok=True)

# Walk through each aircraft class folder
for aircraft in os.listdir(SOURCE):
    aircraft_dir = os.path.join(SOURCE, aircraft)
    img_dir = os.path.join(aircraft_dir, "images")

    # Skip if this class doesn't have an images subfolder
    if not os.path.isdir(img_dir):
        print(f"Warning: Skipping {aircraft} – no 'images' directory found")
        continue

    # Gather all image files (case-insensitive extensions)
    images = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not images:
        print(f"Warning: No images found in {aircraft}")
        continue

    # Shuffle in-place for randomness
    random.shuffle(images)

    # Compute split index
    split_idx = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Create class-specific directories in train/val
    train_class_dir = os.path.join(TARGET, "train", aircraft)
    val_class_dir = os.path.join(TARGET, "val", aircraft)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training images
    for src_path in train_imgs:
        shutil.copy(src_path, train_class_dir)

    # Copy validation images
    for src_path in val_imgs:
        shutil.copy(src_path, val_class_dir)

    print(f"Done: {aircraft} → {len(train_imgs)} train, {len(val_imgs)} val")

print("\nDataset split completed.")