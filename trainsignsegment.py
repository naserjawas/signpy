from torch.utils.data import DataLoader
from datasets.signsegment import SignSegmentDataset
from datasets.collate import collate_fn
from classmodel.temporalcnn import TemporalCNN
from classmodel.bilstm import BiLSTM
import torch.optim as optim
import torch.nn as nn
import torch
import os
import glob

from pathlib import Path


def smoothness_loss(pred):
    # (B, T)
    probs = torch.sigmoid(pred)
    diff = probs[:, 1:] - probs[:, :-1]
    loss = (diff ** 2).mean()
    return loss

def run_epoch(model, loader, criterion, optimiser, device,
              training:bool, lambda_smooth):
    model.train() if training else model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    accuracy = 0
    with torch.set_grad_enabled(training):
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)

            if training:
                optimiser.zero_grad()

            pred = model(x)
            bce = criterion(pred, y)
            smooth = smoothness_loss(pred)
            loss = bce + lambda_smooth * smooth

            if training:
                loss.backward()
                optimiser.step()
            else:
                with torch.no_grad():
                    min_y = y.min()
                    max_y = y.max()
                    percent = 0.8
                    th1 = min_y + percent * (max_y - min_y)
                    th_y = (y >= th1).float()

                    min_pred = pred.min()
                    max_pred = pred.max()
                    th2 = min_pred + percent * (max_pred - min_pred)
                    th_pred = (pred >= th2).float()

                    mask = (th_y == 1.0)
                    pred_ones = th_pred[mask]
                    total += (mask == 1).sum().item()
                    correct += (pred_ones == 1).sum().item()

            total_loss += loss.item()
        if not training:
            accuracy = correct / total
        avg_loss = total_loss / len(loader)

    return avg_loss, accuracy

def get_filelist(dirname):
    dirpath = Path(dirname)
    dirpath = str(dirpath) + os.sep
    filenames = sorted(glob.glob(dirname + "*.npz"))
    print(f"Get {len(filenames)} npz filenames... OK")

    return filenames

if __name__ == "__main__":
    device  = torch.device("mps" if torch.mps.is_available() else "cpu")
    # device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_files = get_filelist("../segprop_train/")
    train_dataset = SignSegmentDataset(train_files)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"DataLoader from {len(train_files)} files... OK")

    validate_files = get_filelist("../segprop_dev/")
    validate_dataset = SignSegmentDataset(validate_files)
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"DataLoader from {len(validate_files)} files... OK")

    # model = TemporalCNN(input_dim=3, hidden_dim=64, num_layers=4)
    model = BiLSTM(input_dim=3, hidden_dim=64, num_layers=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    optimiser = optim.Adam(model.parameters(), lr=1e-6)

    num_epochs = 30
    lambda_smooth = 0.1
    best_acc = 0

    print("Epoch started...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}:")
        avg_loss, acc = run_epoch(model, train_loader, criterion, optimiser,
                                  device, True, lambda_smooth)
        print(f"Training avg. loss: {avg_loss:.4f}")
        avg_loss, acc = run_epoch(model, validate_loader, criterion, optimiser,
                                  device, False, lambda_smooth)
        print(f"Evaluate avg. loss: {avg_loss:.4f}, acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved with best acc: {best_acc:.4f}")
    print(f"Training finised in {num_epochs+1} epoch")
