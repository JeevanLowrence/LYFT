import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix
import seaborn as sns

# ----------------------------
# CONFIG
# ----------------------------
DATASET_ROOT = "aircraft_dataset_final"
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
NUM_WORKERS = 0  # IMPORTANT for Windows
LABEL_SMOOTHING = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFUSION_DIR = "efficient_net_confusion_matrix_plot"
os.makedirs(CONFUSION_DIR, exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes"
)

# ----------------------------
# DATA TRANSFORMS
# ----------------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_tfms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------------
# LOAD DATA
# ----------------------------
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_ROOT, "train"),
    transform=train_tfms
)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_ROOT, "val"),
    transform=val_tfms
)

class_names = train_ds.classes
NUM_CLASSES = len(class_names)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ----------------------------
# CLASS BALANCED LOSS
# ----------------------------
counts = np.bincount(train_ds.targets)
weights = 1.0 / np.maximum(counts, 1)
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)

# ----------------------------
# MODEL
# ----------------------------
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ----------------------------
# METRICS
# ----------------------------
def topk_accuracy(logits, targets, k=5):
    _, pred = logits.topk(k, dim=1)
    return (pred == targets.unsqueeze(1)).any(dim=1).float().mean().item()

# ----------------------------
# TRAIN LOOP
# ----------------------------
best_val_acc = 0.0
all_preds = []
all_labels = []

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
    ):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    top5_scores = []

    with torch.no_grad():
        for imgs, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"
        ):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            top5_scores.append(topk_accuracy(outputs, labels))

            if epoch == EPOCHS - 1:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    top5 = 100 * np.mean(top5_scores)
    gap = train_acc - val_acc

    print(
        f"Epoch {epoch+1} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}% | "
        f"Top-5: {top5:.2f}% | "
        f"Gap: {gap:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "efficientnet_aircraft.pth")

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(18, 16))
sns.heatmap(
    cm,
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
    fmt="d",
    square=True
)
plt.title("EfficientNet-B3 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(CONFUSION_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print("\n✔ Training complete.")
print(f"✔ Best Val Acc: {best_val_acc:.2f}%")
print(f"✔ Confusion matrix saved to: {cm_path}")
print("✔ Class index order:", class_names)
