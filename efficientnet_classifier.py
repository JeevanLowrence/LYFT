import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_ROOT = "aircraft_dataset_final"   # dataset/train, dataset/val
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
IMG_SIZE = 300
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0  # IMPORTANT for Windows
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# FIX PALETTE PNGs
# =========================
def pil_loader_rgb(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =========================
# CLASS-BALANCED LOSS
# =========================
def make_class_weights(dataset):
    counts = np.bincount(dataset.targets)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float)

# =========================
# MODEL
# =========================
def build_model(num_classes):
    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model

# =========================
# METRICS
# =========================
def topk_accuracy(logits, targets, k=5):
    _, pred = logits.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()

# =========================
# MAIN
# =========================
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Datasets
    train_ds = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "train"),
        transform=train_tfms,
        loader=pil_loader_rgb
    )
    val_ds = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "val"),
        transform=val_tfms,
        loader=pil_loader_rgb
    )

    print(f"✔ Classes ({len(train_ds.classes)}): {train_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Model
    model = build_model(len(train_ds.classes)).to(DEVICE)

    # Loss
    class_weights = make_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 0.0
    train_acc_hist, val_acc_hist = [], []

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        correct, total, train_loss = 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_acc_hist.append(train_acc)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []
        top5_scores = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)

                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

                all_preds.extend(out.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                top5_scores.append(topk_accuracy(out, y))

        val_acc = 100 * correct / total
        val_top5 = 100 * np.mean(top5_scores)
        gap = train_acc - val_acc

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(
            f"Epoch {epoch+1} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Top-5: {val_top5:.2f}% | "
            f"Gap: {gap:.2f}%"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "efficientnet_b3_aircraft.pth")

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=train_ds.classes)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("EfficientNet-B3 Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # =========================
    # OVERFITTING DIAGNOSTICS
    # =========================
    plt.figure()
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend()
    plt.title("Overfitting Diagnostic")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()

    print("✔ Training complete. Best Val Acc:", best_val)

# =========================
# ENTRY POINT (WINDOWS SAFE)
# =========================
if __name__ == "__main__":
    main()
