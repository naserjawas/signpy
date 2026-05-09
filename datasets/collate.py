import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys = zip(*batch)

    xs_padded = pad_sequence(xs, batch_first=True)
    ys_padded = pad_sequence(ys, batch_first=True)

    lengths = torch.tensor([x.shape[0] for x in xs])

    return xs_padded, ys_padded, lengths
