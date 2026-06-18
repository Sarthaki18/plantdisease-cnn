"""
Train EfficientNet-B3 on plant disease dataset with MSP + GOAD anomaly detection.
Compatible with app.py inference pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import pickle
import os
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = r"C:\Users\Sarthaki Karnik\Documents\cnn-plantdiseasedetector\New Plant Diseases Dataset(Augmented)\train"
OUTPUT_DIR = Path(__file__).parent / "Classification-based Anomaly Detection"
MODEL_PATH = OUTPUT_DIR / "plant-disease-model (5).pth"
ANOMALY_PARAMS_PATH = OUTPUT_DIR / "anomaly_params (1).pkl"

EPOCHS = 3
BATCH_SIZE = 32
LR = 0.001
NUM_CLASSES = 38
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE (matches app.py)
# ─────────────────────────────────────────────
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.network = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

    def extract_features(self, xb):
        features = self.network.features(xb)
        features = self.network.avgpool(features)
        features = features.flatten(1)
        logits = self.network.classifier(features)
        return features, logits


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        
        print(f"✓ Found {len(self.samples)} images across {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"⚠ Failed to load {img_path}, skipping...")
            return self[np.random.randint(0, len(self))]
        
        if self.transform:
            image = self.transform(image)
        return image, label


# ─────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{total_loss/total:.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), 100 * correct / total


# ─────────────────────────────────────────────
# GOAD TRANSFORMS (matches app.py)
# ─────────────────────────────────────────────
_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
GOAD_TRANSFORMS = [
    transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomRotation((90, 90)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomRotation((180, 180)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomRotation((270, 270)), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _norm]),
    transforms.Compose([transforms.Resize((300, 300)), transforms.RandomRotation((90, 90)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm]),
]
M = len(GOAD_TRANSFORMS)


# ─────────────────────────────────────────────
# ANOMALY DETECTION PARAMETER COMPUTATION
# ─────────────────────────────────────────────
def compute_anomaly_params(model, dataloader, device, epsilon=0.1):
    """Compute GOAD cluster centers and covariance matrix for anomaly detection."""
    print("\n📊 Computing GOAD anomaly detection parameters...")
    
    model.eval()
    cluster_centers = [[] for _ in range(M)]
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            
            for m_idx, transform in enumerate(GOAD_TRANSFORMS):
                # Apply transform to original image batch
                transformed_batch = torch.stack([
                    transform(Image.fromarray((img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
                    for img in images
                ])
                transformed_batch = transformed_batch.to(device)
                
                feat, _ = model.extract_features(transformed_batch)
                cluster_centers[m_idx].append(feat.cpu().numpy())
    
    # Compute cluster centers (mean of features for each transform)
    for i in range(M):
        cluster_centers[i] = np.mean(np.concatenate(cluster_centers[i], axis=0), axis=0)
    
    # Compute covariance matrix from all features
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing covariance", leave=False):
            images = images.to(device)
            feat, _ = model.extract_features(images)
            all_features.append(feat.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    cov_matrix = np.cov(all_features.T)
    
    # Add small regularization to ensure invertibility
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4
    cov_inv = np.linalg.inv(cov_matrix)
    
    # MSP threshold: 50th percentile of max softmax probability
    max_probs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, logits = model.extract_features(images)
            probs = torch.softmax(logits, dim=1)
            max_probs.append(probs.max(dim=1)[0].cpu().numpy())
    
    max_probs = np.concatenate(max_probs)
    msp_threshold = np.percentile(max_probs, 50)
    
    # GOAD threshold: 95th percentile
    goad_scores = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing GOAD scores", leave=False):
            images = images.to(device)
            for m_idx, transform in enumerate(GOAD_TRANSFORMS):
                transformed_batch = torch.stack([
                    transform(Image.fromarray((img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
                    for img in images
                ])
                transformed_batch = transformed_batch.to(device)
                feat, _ = model.extract_features(transformed_batch)
                
                for f in feat.cpu().numpy():
                    mahal = float((f - cluster_centers[m_idx]) @ cov_inv @ (f - cluster_centers[m_idx]))
                    goad_scores.append(mahal)
    
    goad_threshold = np.percentile(goad_scores, 95)
    
    params = {
        'cluster_centers': cluster_centers,
        'cov_inv': cov_inv,
        'msp_threshold': float(msp_threshold),
        'goad_threshold': float(goad_threshold),
        'epsilon': float(epsilon),
    }
    
    print(f"  MSP threshold: {msp_threshold:.4f}")
    print(f"  GOAD threshold: {goad_threshold:.4f}")
    return params


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def main():
    print("🌱 Plant Disease Classification Training Pipeline")
    print(f"📍 Dataset: {DATASET_PATH}")
    print(f"🔧 Device: {DEVICE}")
    print(f"📦 Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving to: {OUTPUT_DIR}")
    
    # Prepare transforms
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = PlantDiseaseDataset(DATASET_PATH, transform=train_transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Update val_dataset transform to remove augmentation
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Initialize model
    print("\n🤖 Initializing EfficientNet-B3...")
    model = EfficientNetB3(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print("\n🚀 Starting training...\n")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"💾 Saving best model (Val Acc: {val_acc:.2f}%)")
            torch.save(model.state_dict(), MODEL_PATH)
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*60}")
    
    # Compute anomaly detection parameters
    print("\n📊 Computing anomaly detection parameters...")
    model.eval()
    
    # Use val_dataset for anomaly params
    val_loader_no_aug = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    anomaly_params = compute_anomaly_params(model, val_loader_no_aug, DEVICE)
    
    # Save anomaly params
    with open(ANOMALY_PARAMS_PATH, 'wb') as f:
        pickle.dump(anomaly_params, f)
    print(f"✅ Anomaly parameters saved to {ANOMALY_PARAMS_PATH}")
    
    print("\n🎉 Training complete! Model ready for inference via app.py")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Params: {ANOMALY_PARAMS_PATH}")


if __name__ == "__main__":
    main()
