"""
Simplified training script for plant disease classification.
This version is optimized for quick setup and execution.
"""

import sys

# Import and run training
try:
    import torch
except ImportError:
    print("❌ PyTorch not installed. Please install it first:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.optim as optim
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


# CONFIG
DATASET_PATH = r"C:\Users\Sarthaki Karnik\Documents\cnn-plantdiseasedetector\New Plant Diseases Dataset(Augmented)\train"
OUTPUT_DIR = Path(__file__).parent / "Classification-based Anomaly Detection"
MODEL_PATH = OUTPUT_DIR / "plant-disease-model (5).pth"
ANOMALY_PARAMS_PATH = OUTPUT_DIR / "anomaly_params (1).pkl"

EPOCHS = 3
BATCH_SIZE = 32
LR = 0.001
NUM_CLASSES = 38
DEVICE = torch.device('cpu')  # Force CPU
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# MODEL
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


# DATASET
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
            return self[np.random.randint(0, len(self))]
        
        if self.transform:
            image = self.transform(image)
        return image, label


# TRAINING
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
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


# MAIN
def main():
    print("🌱 Plant Disease Classification Training Pipeline")
    print(f"📍 Dataset: {DATASET_PATH}")
    print(f"🔧 Device: {DEVICE}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n📂 Loading dataset...")
    dataset = PlantDiseaseDataset(DATASET_PATH, transform=train_transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    print("\n🤖 Initializing EfficientNet-B3...")
    model = EfficientNetB3(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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
    print(f"Model saved to: {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
