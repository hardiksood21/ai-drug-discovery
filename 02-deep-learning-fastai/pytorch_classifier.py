"""
Phase 2 Deep Learning Baseline: PyTorch Neural Network & Transfer Learning
-------------------------------------------------------------------------
Architecture: ResNet-18 (torchvision.models)
Framework: PyTorch (torch, torchvision)
Optimization: AdamW + Cosine Annealing Learning Rate Scheduler
Evaluation: Validation Accuracy, CrossEntropy Loss, Classification Report
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    return device

def build_model(num_classes=10):
    """Load pre-trained ResNet-18 and replace final classification head."""
    print("Initializing ResNet-18 backbone...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze early convolutional layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def main():
    device = setup_device()
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 benchmark dataset
    print("Loading benchmark dataset (CIFAR-10)...")
    data_dir = "./data"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Build Model
    model = build_model(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    epochs = 5
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\nStarting PyTorch Model Training Loop...")
    print("=" * 55)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    print("=" * 55)
    print(f"Final Validation Accuracy: {val_accs[-1]*100:.2f}%")
    
    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(1, epochs+1), train_losses, label="Train Loss", color="#3B82F6", marker='o')
    ax1.plot(range(1, epochs+1), val_losses, label="Val Loss", color="#EF4444", marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.plot(range(1, epochs+1), [a*100 for a in train_accs], label="Train Acc", color="#3B82F6", marker='o')
    ax2.plot(range(1, epochs+1), [a*100 for a in val_accs], label="Val Acc", color="#10B981", marker='^')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("pytorch_training_curves.png", dpi=300)
    print("Saved training metrics plot to 'pytorch_training_curves.png'.")

if __name__ == "__main__":
    main()
