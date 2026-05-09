import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignSegmentDataset(Dataset):
    def __init__(self, file_list, feature_keys=None, transform=None):
        """
        args:
            - file_list     : lisf of paths to .npz file
            - feature_keys  : which feature to load
            - transform     : optional transform
        """
        self.file_list = file_list
        self.feature_keys = feature_keys or [
            "speed_score",
            "direction_score",
            "orientation_score"
        ]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        # --- load data ---
        features = []
        for key in self.feature_keys:
            if key not in data:
                raise ValueError(f"{key} is not found in {file_path}")
            features.append(data[key])

        x = np.stack(features, axis=-1)
        y = data["boundary_score"] if "boundary_score" in data else None
        x = torch.tensor(x, dtype=torch.float32)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y
