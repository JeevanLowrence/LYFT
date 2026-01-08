import os
import shutil
import random
from collections import defaultdict

# =========================
# CONFIG
# =========================
DATASET_ROOT = "aircraft_dataset"
OUTPUT_ROOT = "aircraft_dataset_final"

TRAIN_RATIO = 0.8
MIN_IMAGES_PER_CLASS = 30
SEED = 42

random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# =========================
# AIRCRAFT FAMILY MERGES
# =========================
FAMILY_MAP = {
    # Fighters
    "F-15 Eagle": "F-15",
    "McDonnell Douglas F-15 Eagle": "F-15",

    "F-16 Fighting Falcon": "F-16",
    "General Dynamics F-16 Fighting Falcon": "F-16",

    "F-22 Raptor": "F-22",
    "Lockheed Martin F-22 Raptor": "F-22",

    "F-35B Lightning II": "F-35",

    "Saab JAS 39 Gripen": "JAS 39 Gripen",

    "Northrop F-5": "F-5",
    "Northrop F-5 Freedom Fighter": "F-5",

    "Sukhoi Su-30": "Su-30",
    "Sukhoi Su-30MKI": "Su-30",

    "Dassault Mirage F1": "Mirage F1",

    "Messerschmitt Bf 109": "Bf 109",

    "Supermarine Spitfire": "Spitfire",

    "Mitsubishi A6M": "A6M Zero",
    "Mitsubishi A6M Zero": "A6M Zero",

    # Bombers
    "B-52 Stratofortress": "B-52",
    "Rockwell B-1 Lancer": "B-1",

    # Transport / Utility
    "Lockheed C-130 Hercules": "C-130",
    "Lockheed C-130H Hercules": "C-130",

    "Douglas DC-3": "DC-3",

    "Antonov An-2": "An-2",
    "Antonov An-72": "An-72",

    # Helicopters
    "CH-47 Chinook": "CH-47",
    "Chinook": "CH-47",

    "Mil Mi-8": "Mi-8",
    "Westland Sea King": "Sea King",
    "HR3S Sea King": "Sea King",

    # UAVs (kept as physical aircraft)
    "General Atomics MQ-9 Reaper": "MQ-9",
    "MQ-9 Reaper": "MQ-9",

    "Northrop Grumman RQ-4 Global Hawk": "RQ-4",
    "Schiebel Camcopter S-100": "Camcopter S-100",
    "XQ-58": "XQ-58 Valkyrie",
    "Valiant-XQ58": "XQ-58 Valkyrie",

    # Trainers / misc
    "Pilatus PC-21": "PC-21",
    "Northrop T-38 Talon": "T-38",
    "Northrop T-38A Talon": "T-38",

    # Others
    "Junkers Ju 52": "Ju 52",
    "Dornier 228": "Dornier 228",
    "Dornier Do 28": "Do 28",
    "Tupolev Tu-154": "Tu-154",
}

# =========================
# DROP RULES (HARD)
# =========================
DROP_EXACT = {
    # Countries
    "Brazil", "Canada", "France", "Germany", "Greece",
    "Japan", "Spain", "Thailand", "US",

    # Roles / concepts
    "Attack", "Air superiority", "Special Operations",
    "Tanker", "Delivery Drone",

    # Properties
    "Supersonic", "VTOL aircraft",

    # Generic categories
    "UAV", "UCAV", "Unmanned", "Quadcopter", "Gliders",

    # Garbage
    "name", "E", "H", "J", "T", "Z",
}

# =========================
# UTILS
# =========================
def is_image(fname):
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def normalize_class(name):
    name = name.strip()
    return FAMILY_MAP.get(name, name)


def should_drop(name):
    return name in DROP_EXACT


def collect_images(path):
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if is_image(f)
    ]


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# MAIN
# =========================
def main():
    print("🔍 Building clean aircraft dataset...")

    merged = defaultdict(list)

    for folder in os.listdir(DATASET_ROOT):
        img_dir = os.path.join(DATASET_ROOT, folder, "images")
        if not os.path.isdir(img_dir):
            continue

        images = collect_images(img_dir)
        if not images:
            continue

        canonical = normalize_class(folder)

        if should_drop(canonical):
            continue

        merged[canonical].extend(images)

    # Drop weak classes
    final_classes = {
        cls: imgs
        for cls, imgs in merged.items()
        if len(imgs) >= MIN_IMAGES_PER_CLASS
    }

    print(f"✅ Final aircraft classes: {len(final_classes)}")

    # Rebuild output
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    train_root = os.path.join(OUTPUT_ROOT, "train")
    val_root = os.path.join(OUTPUT_ROOT, "val")

    for cls, imgs in sorted(final_classes.items()):
        random.shuffle(imgs)
        split = int(len(imgs) * TRAIN_RATIO)

        train_imgs = imgs[:split]
        val_imgs = imgs[split:]

        train_dir = os.path.join(train_root, cls)
        val_dir = os.path.join(val_root, cls)

        safe_mkdir(train_dir)
        safe_mkdir(val_dir)

        for img in train_imgs:
            shutil.copy2(img, train_dir)

        for img in val_imgs:
            shutil.copy2(img, val_dir)

        print(f"{cls:20s}  train={len(train_imgs):4d}  val={len(val_imgs):4d}")

    print("\n🎉 Dataset ready")
    print(f"📁 Output: {OUTPUT_ROOT}")
    print("\n📋 Class list:")
    for cls in sorted(final_classes):
        print(f" - {cls}")


if __name__ == "__main__":
    main()
