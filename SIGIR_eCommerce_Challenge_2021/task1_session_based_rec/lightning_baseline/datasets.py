import pandas as pd
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SessionDataset(Dataset):
    """Dataset of (input_sequence, target_item) pairs."""
    def __init__(self, sessions: List[List[int]], item2idx: Dict[int, int], max_len: int = 20) -> None:
        self.inputs = []
        self.targets = []
        self.max_len = max_len
        for s in sessions:
            idxs = [item2idx[i] for i in s if i in item2idx]
            if len(idxs) < 2:
                continue
            self.inputs.append(idxs[:-1][-max_len:])
            self.targets.append(idxs[-1])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]

def collate_batch(batch):
    seqs, targets = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = [([0] * (max_len - len(s)) + s) for s in seqs]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

class SessionDataModule(pl.LightningDataModule):
    def __init__(self, train_path: Path, session_col: str = "session_id", item_col: str = "product_id", batch_size: int = 64):
        super().__init__()
        self.train_path = train_path
        self.session_col = session_col
        self.item_col = item_col
        self.batch_size = batch_size
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.dataset: SessionDataset | None = None

    def setup(self, stage: str | None = None):
        df = pd.read_csv(self.train_path)
        sessions = df.groupby(self.session_col)[self.item_col].apply(list).tolist()
        unique_items = sorted({i for s in sessions for i in s})
        self.item2idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.dataset = SessionDataset(sessions, self.item2idx)

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)
