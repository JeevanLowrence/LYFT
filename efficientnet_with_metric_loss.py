import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_ROOT = "aircraft_dataset_final"
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
IMG_SIZE = 300
EMBED_DIM = 512
NUM_WORKERS = 0  # Windows safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# IMAGE LOADER (FIX PNG)
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
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# =========================
# MODEL
# =========================
class EfficientNetMetric(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        base = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(1536, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.embed(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# =========================
# TRIPLET LOSS (BATCH HARD)
# =========================
def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    embeddings: (B, D) normalized
    labels: (B,)
    """
    device = embeddings.device
    dist = torch.cdist(embeddings, embeddings, p=2)

    losses = []

    for i in range(len(labels)):
        pos_mask = labels == labels[i]
        neg_mask = labels != labels[i]

        pos_mask[i] = False  # exclude self-distance

        if pos_mask.any() and neg_mask.any():
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            losses.append(F.relu(hardest_pos - hardest_neg + margin))

    if len(losses) == 0:
        # return zero tensor WITH gradient
        return torch.zeros(1, device=device, requires_grad=True)

    return torch.stack(losses).mean()
    

# =========================
# MAIN
# =========================
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

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

    model = EfficientNetMetric(EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            emb = model(x)
            loss = batch_hard_triplet_loss(emb, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} - Triplet Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), "aircraft_metric_model.pth")
    print("✔ Metric model saved")

    # =========================
    # EVALUATION VIA k-NN
    # =========================
    model.eval()
    def extract_embeddings(loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                emb = model(x).cpu().numpy()
                feats.append(emb)
                labels.append(y.numpy())
        return np.vstack(feats), np.concatenate(labels)

    train_X, train_y = extract_embeddings(train_loader)
    val_X, val_y = extract_embeddings(val_loader)

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(train_X, train_y)
    preds = knn.predict(val_X)

    acc = accuracy_score(val_y, preds) * 100
    print(f"✔ k-NN Validation Accuracy: {acc:.2f}%")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
