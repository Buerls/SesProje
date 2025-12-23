import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
import random
import math
from dataset import AudioDataset
from model import LanguageTCN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_processed.csv")
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3


def apply_spec_augment(spec, prob=0.5):
    if random.random() > prob:
        return spec

    f = random.randint(0, 20)
    f0 = random.randint(0, 128 - f)
    spec[:, f0:f0 + f, :] = 0

    t = random.randint(0, 40)
    t0 = random.randint(0, 157 - t)
    spec[:, :, t0:t0 + t] = 0
    return spec


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TCN Optimized | Cihaz: {device}")

    full_dataset = AudioDataset(CSV_PATH)
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = LanguageTCN(num_inputs=128, num_channels=[64, 128, 256, 512], num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = apply_spec_augment(inputs, prob=0.6)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("⚠️ Loss NaN oldu! Batch atlanıyor...")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            scheduler.step(epoch + 1 / len(train_loader))

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        elapsed = time.time() - start_time

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] ({elapsed:.0f}s) LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train: %{train_acc:.2f} | Test: %{test_acc:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_model_pro.pth"))
            print(f"     REKOR! (%{test_acc:.2f}) -> Kaydedildi.")

    print(f"\n🏁 FİNAL BAŞARI: %{best_acc:.2f}")


if __name__ == "__main__":
    train()
