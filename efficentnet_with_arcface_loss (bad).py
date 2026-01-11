import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from tqdm import tqdm
import numpy as np
from PIL import Image


# ---------------- CONFIG ----------------
DATASET_ROOT = "aircraft_dataset_final"  # dataset/train , dataset/val
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
EMBED_DIM = 512
ARC_MARGIN = 0.5
ARC_SCALE = 30.0
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0  # Windows safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pil_loader_rgb(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")

# ---------------------------------------


# ---------------- ARCFACE HEAD ----------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


# ---------------- MODEL ----------------
class EfficientNetArcFace(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.embedding = nn.Linear(in_features, EMBED_DIM)
        self.bn = nn.BatchNorm1d(EMBED_DIM)
        self.arcface = ArcMarginProduct(EMBED_DIM, num_classes, ARC_SCALE, ARC_MARGIN)

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        emb = self.bn(self.embedding(feats))

        if labels is not None:
            logits = self.arcface(emb, labels)
            return logits, emb
        return emb


# ---------------- DATA ----------------
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "train"),
        transform=train_tf,
        loader=pil_loader_rgb
    )

    val_ds = datasets.ImageFolder(
        os.path.join(DATASET_ROOT, "val"),
        transform=val_tf,
        loader=pil_loader_rgb
    )


    class_counts = np.bincount(train_ds.targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_ds.targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, train_ds.classes


# ---------------- METRICS ----------------
def accuracy(outputs, targets, topk=(1, 5)):
    maxk = max(topk)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / targets.size(0)) * 100)
    return res


# ---------------- TRAIN ----------------
def main():
    train_loader, val_loader, class_names = get_loaders()
    num_classes = len(class_names)

    model = EfficientNetArcFace(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val = 0.0

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_correct, train_total = 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(imgs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # ---- VALIDATE ----
        model.eval()
        val_correct, val_total = 0, 0
        top5_sum = 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                emb = model(imgs)
                logits = F.linear(F.normalize(emb),
                                  F.normalize(model.arcface.weight)) * ARC_SCALE

                acc1, acc5 = accuracy(logits, labels)
                top5_sum += acc5.item() * labels.size(0)

                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        top5 = top5_sum / val_total
        gap = train_acc - val_acc

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Top-5: {top5:.2f}% | Gap: {gap:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "efficientnet_arcface.pth")

    print(f"✔ Training complete. Best Val Acc: {best_val:.2f}%")
    print("✔ Class index order:", class_names)


if __name__ == "__main__":
    main()
