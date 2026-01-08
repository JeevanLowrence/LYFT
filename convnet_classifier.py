import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# CONFIG
# =========================
DATA_ROOT = "aircraft_dataset_final"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_SMOOTHING = 0.1
MODEL_OUT = "aircraft_classifier_balanced.pth"

# =========================
# SAFE DATASET (NO PIL WARNINGS)
# =========================
class SafeImageFolder(datasets.ImageFolder):
    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            if img.mode == "P":
                img = img.convert("RGBA")
            return img.convert("RGB")

# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =========================
# CLASS-BALANCED LOSS
# =========================
def compute_class_weights(dataset, beta=0.9999):
    counts = np.bincount(dataset.targets)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

# =========================
# MODEL
# =========================
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# =========================
# MAIN
# =========================
def main():
    train_ds = SafeImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = SafeImageFolder(VAL_DIR, transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)

    print(f"\n✔ Classes ({num_classes}):")
    print(class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(num_classes).to(DEVICE)

    class_weights = compute_class_weights(train_ds).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    train_acc_hist, val_acc_hist = [], []

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item()

        train_acc = 100 * correct / total
        train_acc_hist.append(train_acc)

        print(f"Epoch {epoch+1} - Train Loss: {loss_sum/len(train_loader):.3f}")
        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.2f}%")

        # -------- VALIDATION --------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = 100 * correct / total
        val_acc_hist.append(val_acc)

        gap = train_acc - val_acc
        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.2f}% | Gap: {gap:.2f}%")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\n✔ Model saved as {MODEL_OUT}")

    # =========================
    # OVERFITTING DIAGNOSTICS
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # CONFUSION MATRIX + TOP-5
    # =========================
    y_true, y_pred = [], []
    top5_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            _, top5 = out.topk(5, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())

            for i in range(y.size(0)):
                if y[i].item() in top5[i]:
                    top5_correct += 1
                total += 1

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(include_values=False, cmap="viridis", ax=ax, xticks_rotation=90)
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print(f"\n✔ Top-5 Accuracy: {100 * top5_correct / total:.2f}%")

# =========================
# ENTRY POINT (CRITICAL)
# =========================
if __name__ == "__main__":
    main()
