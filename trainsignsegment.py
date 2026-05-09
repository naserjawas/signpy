from torch.utils.data import DataLoader
from datasets.signsegment import SignSegmentDataset
from datasets.collate import collate_fn
from classmodel.temporalcnn import TemporalCNN
from classmodel.bilstm import BiLSTM
import torch.optim as optim
import torch.nn as nn
import torch


def smoothness_loss(pred):
    # (B, T)
    probs = torch.sigmoid(pred)
    diff = probs[:, 1:] - probs[:, :-1]
    loss = (diff ** 2).mean()
    return loss

device  = torch.device("mps" if torch.mps.is_available() else "cpu")
# device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

file_list = [
    "../segprop/01April_2010_Thursday_heute_default-0_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-2_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-3_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-4_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-6_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-7_segprop.npz",
    "../segprop/01April_2010_Thursday_heute_default-8_segprop.npz",
]

dataset = SignSegmentDataset(file_list)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


model = TemporalCNN(input_dim=3, hidden_dim=64, num_layers=4)
# model = BiLSTM(input_dim=3, hidden_dim=64, num_layers=1)
model = model.to(device)

optimiser = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30
lambda_smooth = 0.1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        y = y.float()

        optimiser.zero_grad()

        pred = model(x)
        bce = criterion(pred, y)
        smooth = smoothness_loss(pred)
        loss = bce + lambda_smooth * smooth

        loss.backward()
        optimiser.step()
        total_loss += loss.item()

        print(pred.min().item(), pred.max().item())
        print(loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Avg. Loss: {avg_loss:.4f}")

    # model.eval()


