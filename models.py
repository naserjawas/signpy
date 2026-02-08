import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class VideoFrameDataset(Dataset):
    def __init__(self, dataset_dict, video_list):
        self.dataset = dataset_dict
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        features = self.dataset[video_name]["features"]
        labels = self.dataset[video_name]["labels"]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return features, labels

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        out = self.dropout(out)

        return x + out

class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, in_dim, num_classes):
        super().__init__()

        self.conv_in = nn.Conv1d(in_dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2**i, num_f_maps, num_f_maps)
            for i in range(num_layers)]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)

        return out

class MSTCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, in_dim, num_classes):
        super().__init__()

        self.stage1 = SingleStageTCN(num_layers, num_f_maps, in_dim, num_classes)

        self.stages = nn.ModuleList(
            [SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes)
            for _ in range(num_stages - 1)]
        )

    def forward(self, x):
        out = self.stage1(x)

        outputs = [out]

        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)

        return outputs
