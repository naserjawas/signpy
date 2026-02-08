import os
import glob
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from models import VideoFrameDataset
from models import MSTCN
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def load_npz_files(datapath):
    dirname = str(datapath) + os.sep
    filenames = sorted(glob.glob(dirname + "*.npz"))

    return filenames

def construct_segment(listgtdata):
    frmsegment = []
    last = 0
    for i, g in enumerate(listgtdata):
        if i == 0:
            frmsegment.append(0)
            last = g
        else:
            if ((g not in [last, last+1, last+2]) or
                (g == last and  listgtdata[i-1] == last+2)):
                frmsegment[-1] = 1
                frmsegment.append(1)
                last = g
            else:
                frmsegment.append(0)

    return frmsegment

def temporal_smoothness_loss(pred):
    return torch.mean(torch.abs(pred[:, :, 1:] - pred[:, :, :-1]))

def get_boundaries(frame_labels):
    return np.where(frame_labels == 1)[0]

def boundary_f1_score(gt_labels, pred_labels, tolerance=5):
    gt_boundaries = get_boundaries(gt_labels)
    pred_boundaries = get_boundaries(pred_labels)

    matched_gt = set()
    matched_pred = set()

    for i, p in enumerate(pred_boundaries):
        for j, g in enumerate(gt_boundaries):
            if j in matched_gt:
                continue
            if abs(p - g) <= tolerance:
                matched_gt.add(j)
                matched_pred.add(i)
                break
    TP = len(matched_pred)
    FP = len(pred_boundaries) - TP
    FN = len(gt_boundaries) - len(matched_gt)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1, precision, recall


if __name__ == '__main__':
    datapath = Path("../results/")
    npzfiles = load_npz_files(datapath)

    # construct the dataset
    dataset = {}
    for npzf in npzfiles:
        data = np.load(npzf)
        filename = npzf.split(os.sep)[-1]
        vidname = filename[:-4]
        # summag = data['summag']
        summag_r = data['summag_r']
        summag_l = data['summag_l']
        # lspose_r = data['lspose_r']
        # lspose_l = data['lspose_l']
        # frmnum = data['frmnum']
        gtdata = data['gtdata']

        frmsegment = construct_segment(gtdata)

        features = np.stack((summag_r, summag_l), axis=1)
        labels = np.array(frmsegment, dtype=np.uint8)

        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

        contents = {}
        contents["features"] = features
        contents["labels"] = labels
        dataset[vidname] = contents

    # split to train and test
    video_names = list(dataset.keys())
    random.shuffle(video_names)
    split_idx = int(len(video_names) * 0.8)

    train_videos = video_names[:split_idx]
    test_videos = video_names[split_idx:]

    train_dataset = VideoFrameDataset(dataset, train_videos)
    test_dataset = VideoFrameDataset(dataset, test_videos)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Train
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MSTCN(
            num_stages=3,
            num_layers=5,
            num_f_maps=64,
            in_dim=2,
            num_classes=2).to(device)
    weights = torch.tensor([1.0, 30.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epoch = 30
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            features = features.permute(0, 2, 1)
            optimiser.zero_grad()
            outputs = model(features)

            loss = torch.tensor(0.0, device=device)
            for out in outputs:
                ce = criterion(out, labels)
                smooth = temporal_smoothness_loss(torch.softmax(out, dim=1))
                loss += ce #+ 0.15 * smooth

            loss = loss / len(outputs)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        total_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    # Test
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            features = features.permute(0, 2, 1)

            outputs = model(features)
            out = outputs[-1]

            pred = torch.argmax(out, dim=1)

            gt_np = labels.cpu().numpy().flatten()
            pred_np = pred.cpu().numpy().flatten()

            f1, p, r = boundary_f1_score(gt_np, pred_np, tolerance=5)
            all_f1.append(f1)

    print(f"Boundary F1@5: {np.mean(all_f1):.4f}")
