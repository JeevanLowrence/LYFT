import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "aircraft_dataset_final"
EPOCHS = 40
LR = 1e-4
BATCH_SIZE = 8
IMG_SIZE = 384


# -----------------------------
# DATASET SANITY FIX
# -----------------------------
def remove_empty_class_dirs(root):
    """
    ImageFolder crashes if class folders exist but contain no images.
    This function removes those folders before dataset loading.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    for split in ["train", "val"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue

        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            images = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(valid_exts)
            ]

            if len(images) == 0:
                print(f"🧹 Removing empty class folder: {split}/{cls}")
                os.rmdir(cls_dir)


# -----------------------------
# MAIN
# -----------------------------
def main():

    print(f"🚀 Using device: {DEVICE}")

    # Fix dataset before ImageFolder touches it
    remove_empty_class_dirs(DATA_DIR)

    # -----------------------------
    # TRANSFORMS
    # -----------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # -----------------------------
    # DATASET
    # -----------------------------
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    # Hard safety checks
    assert len(train_ds) > 0, "❌ Training dataset is empty"
    assert len(val_ds) > 0, "❌ Validation dataset is empty"

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    num_classes = len(train_ds.classes)
    print(f"✅ Classes detected ({num_classes}):")
    for c in train_ds.classes:
        print("   -", c)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
            colour="green"
        )

        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss:.3f}")

        print(f"Epoch {epoch+1} - Train Loss: {running_loss:.3f}")

        # ---- VALIDATE ----
        model.eval()
        correct = 0
        total = 0

        vbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
            colour="blue"
        )

        with torch.no_grad():
            for images, labels in vbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                acc = 100.0 * correct / total
                vbar.set_postfix(acc=f"{acc:.2f}%")

        print(f"Epoch {epoch+1} - Val Accuracy: {acc:.2f}%")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    torch.save(model.state_dict(), "aircraft_classifier.pth")
    print("\n✔ Model saved as aircraft_classifier.pth")
    print("✔ Class index order:", train_ds.classes)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()


