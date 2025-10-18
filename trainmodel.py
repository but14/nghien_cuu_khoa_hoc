# train_classifier.py
import os, json, argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)   # OUT_DIR
    p.add_argument("--save_path", type=str, default="face_classifier.pth")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def build_model(num_classes, device, feature_extract=False):
    # pretrained ResNet18 -> change final FC
    model = models.resnet18(pretrained=True)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model.to(device)

def train(data_dir, save_path, epochs, batch_size, lr, img_size, device):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=val_tf)

    # Save label map
    idx2class = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(Path(save_path).with_suffix(".labels.json"), "w", encoding="utf-8") as f:
        json.dump(idx2class, f, ensure_ascii=False, indent=2)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_ds.classes)
    print(f"[INFO] classes: {num_classes}")

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for imgs, labels in tqdm(train_loader, desc=f"Train E{epoch}"):
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
        epoch_loss = running_loss / len(train_ds)
        epoch_acc  = running_corrects / len(train_ds)

        # validation
        model.eval()
        val_corrects = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += (outputs.argmax(dim=1) == labels).sum().item()
        val_loss = val_loss / len(val_ds)
        val_acc  = val_corrects / len(val_ds)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "img_size": img_size
            }, save_path)
            print(f"[INFO] Saved best model (val_acc={best_acc:.4f}) -> {save_path}")

    print("[DONE] Training finished. Best val_acc:", best_acc)

if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.save_path, args.epochs, args.batch_size, args.lr, args.img_size, args.device)
