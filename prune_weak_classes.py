import os
import shutil

DATASET_ROOT = "dataset"
MIN_IMAGES = 20
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def count_images(folder):
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTS)
    )


def main():
    train_dir = os.path.join(DATASET_ROOT, "train")
    val_dir = os.path.join(DATASET_ROOT, "val")

    removed_classes = []

    # -----------------------------
    # PRUNE BASED ON TRAIN ONLY
    # -----------------------------
    for cls in os.listdir(train_dir):
        cls_train_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_train_dir):
            continue

        n_train = count_images(cls_train_dir)

        if n_train < MIN_IMAGES:
            print(f"❌ Removing train/{cls} ({n_train} images)")
            shutil.rmtree(cls_train_dir)
            removed_classes.append(cls)

    # -----------------------------
    # REMOVE VAL ONLY IF TRAIN REMOVED
    # -----------------------------
    for cls in removed_classes:
        cls_val_dir = os.path.join(val_dir, cls)
        if os.path.isdir(cls_val_dir):
            print(f"🧹 Removing val/{cls} (train removed)")
            shutil.rmtree(cls_val_dir)

    print(f"\n✔ Removed {len(removed_classes)} weak classes (< {MIN_IMAGES} images)")
    print("✔ Validation kept in sync safely")


if __name__ == "__main__":
    main()
