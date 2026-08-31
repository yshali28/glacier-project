# ================================================================
# train.py
# Full training pipeline for Attention U-Net
# Run on Kaggle with GPU T4 enabled
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

from model import AttentionUNet

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")  # if running on Kaggle/Colab, point this at your uploaded dataset instead
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS        = 30
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4

print(f"Using device: {DEVICE}")


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------
class GlacierDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        """
        images: numpy array (N, H, W, 3)  — NDSI, NDWI, NDVI channels
        masks:  numpy array (N, H, W)     — binary glacier mask
        """
        self.images  = images
        self.masks   = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask  = self.masks[idx].copy()

        # Data augmentation (only during training)
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask  = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask  = np.flipud(mask).copy()
            if np.random.rand() > 0.5:
                k     = np.random.choice([1, 2, 3])
                image = np.rot90(image, k).copy()
                mask  = np.rot90(mask,  k).copy()

        # (H, W, 3) -> (3, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        # (H, W)   -> (1, H, W)
        mask  = torch.tensor(mask,  dtype=torch.float32).unsqueeze(0)

        return image, mask


# ----------------------------------------------------------------
# Loss Function: Dice Loss + BCE
# Better than BCE alone for imbalanced segmentation tasks
# ----------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def dice_loss(self, pred, target):
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)


# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------
def compute_iou(pred, target, threshold=0.5):
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()


def compute_dice(pred, target, threshold=0.5):
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2 * intersection / (pred_bin.sum() + target.sum() + 1e-8)).item()


def compute_precision_recall(pred, target, threshold=0.5):
    pred_bin  = (pred > threshold).float()
    tp        = (pred_bin * target).sum()
    fp        = (pred_bin * (1 - target)).sum()
    fn        = ((1 - pred_bin) * target).sum()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall    = (tp / (tp + fn + 1e-8)).item()
    return precision, recall


# ----------------------------------------------------------------
# Training and Validation loops
# ----------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou, total_dice = 0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(outputs.detach(), masks)
        total_dice += compute_dice(outputs.detach(), masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_dice = 0, 0, 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)

            total_loss += loss.item()
            total_iou  += compute_iou(outputs, masks)
            total_dice += compute_dice(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
# Load data
X_train = np.load(PROCESSED_DIR / "X_train.npy")
X_val   = np.load(PROCESSED_DIR / "X_val.npy")
X_test  = np.load(PROCESSED_DIR / "X_test.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_val   = np.load(PROCESSED_DIR / "y_val.npy")
y_test  = np.load(PROCESSED_DIR / "y_test.npy")

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# Datasets and loaders
train_ds     = GlacierDataset(X_train, y_train, augment=True)
val_ds       = GlacierDataset(X_val,   y_val,   augment=False)
test_ds      = GlacierDataset(X_test,  y_test,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model, loss, optimizer, scheduler
model     = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
best_val_iou = 0
history = {
    "train_loss": [], "val_loss": [],
    "train_iou":  [], "val_iou":  [],
    "train_dice": [], "val_dice": []
}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_iou, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_iou, vl_dice = validate(model, val_loader, criterion, DEVICE)
    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_iou"].append(tr_iou)
    history["val_iou"].append(vl_iou)
    history["train_dice"].append(tr_dice)
    history["val_dice"].append(vl_dice)

    if vl_iou > best_val_iou:
        best_val_iou = vl_iou
        torch.save(model.state_dict(), "best_attention_unet.pth")

    if epoch % 1 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f} | "
              f"IoU: {tr_iou:.4f}/{vl_iou:.4f} | "
              f"Dice: {tr_dice:.4f}/{vl_dice:.4f}")

# Final test evaluation
model.load_state_dict(torch.load("best_attention_unet.pth"))
test_loss, test_iou, test_dice = validate(model, test_loader, criterion, DEVICE)

all_prec, all_rec = [], []
model.eval()
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        p, r    = compute_precision_recall(outputs, masks)
        all_prec.append(p)
        all_rec.append(r)

precision = np.mean(all_prec)
recall    = np.mean(all_rec)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print("\n" + "=" * 50)
print("ATTENTION U-NET -- TEST SET RESULTS")
print("=" * 50)
print(f"IoU (Jaccard):   {test_iou:.4f}")
print(f"Dice Score:      {test_dice:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"F1 Score:        {f1:.4f}")
print("=" * 50)

# Training curves plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"],   label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].set_xlabel("Epoch")

axes[1].plot(history["train_iou"], label="Train IoU")
axes[1].plot(history["val_iou"],   label="Val IoU")
axes[1].set_title("IoU Curve")
axes[1].legend()
axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")
