"""Torch Dataset of padded sequences and next-item targets."""

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_left(seq, max_len):
    seq = list(seq)[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    if seq:
        out[-len(seq):] = seq
    return out


class TrainSequences(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = [seq for seq in sequences if len(seq) > 1]
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        inputs = pad_left(seq[:-1], self.max_len)
        targets = pad_left(seq[1:], self.max_len)
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def sequence_batch(histories, users, max_len):
    return torch.from_numpy(np.stack([pad_left(histories[u], max_len) for u in users]))
