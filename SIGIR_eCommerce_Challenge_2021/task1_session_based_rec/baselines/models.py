from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class BaseModel:
    """Common interface for baseline models."""

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def predict_sessions(self, sessions: List[List[int]], top_k: int) -> List[List[int]]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        raise NotImplementedError


class PopularityModel(BaseModel):
    """Simple popularity based model."""

    def __init__(self, popularity: pd.Series | None = None) -> None:
        self.popularity = popularity if popularity is not None else pd.Series(dtype=int)

    def train(self, train_path: Path, item_col: str = "product_id") -> None:
        df = pd.read_csv(train_path)
        self.popularity = df[item_col].value_counts()

    def predict_sessions(self, sessions: List[List[int]], top_k: int) -> List[List[int]]:
        top_items = self.popularity.sort_values(ascending=False).head(top_k).index.tolist()
        return [top_items for _ in sessions]

    def save(self, path: Path) -> None:
        self.popularity.to_csv(path, header=False)

    @classmethod
    def load(cls, path: Path) -> "PopularityModel":
        pop = pd.read_csv(path, header=None, names=["item", "count"])
        return cls(pop.set_index("item")["count"])


class SessionDataset(Dataset):
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


class GRURecModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))


class RNNRecModel(BaseModel):
    """Simple GRU based next item predictor."""

    def __init__(self, model: GRURecModel | None = None, item2idx: Dict[int, int] | None = None) -> None:
        self.model = model
        self.item2idx = item2idx or {}
        self.idx2item = {v: k for k, v in (item2idx or {}).items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)

    def train(
        self,
        train_path: Path,
        session_col: str = "session_id",
        item_col: str = "product_id",
        epochs: int = 5,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        df = pd.read_csv(train_path)
        sessions = df.groupby(session_col)[item_col].apply(list).tolist()
        unique_items = sorted({i for s in sessions for i in s})
        self.item2idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        dataset = SessionDataset(sessions, self.item2idx)
        self.model = GRURecModel(len(self.item2idx) + 1, embedding_dim, hidden_dim).to(self.device)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self.model.parameters())
        self.model.train()
        for _ in range(epochs):
            for seq, target in loader:
                seq = seq.to(self.device)
                target = target.to(self.device)
                optim.zero_grad()
                out = self.model(seq)
                loss = criterion(out, target)
                loss.backward()
                optim.step()

    def predict_sessions(self, sessions: List[List[int]], top_k: int) -> List[List[int]]:
        self.model.eval()
        preds: List[List[int]] = []
        with torch.no_grad():
            for s in sessions:
                idxs = [self.item2idx.get(i, 0) for i in s][-20:]
                seq = torch.tensor([idxs], dtype=torch.long, device=self.device)
                logits = self.model(seq)
                top = logits.topk(top_k, dim=1).indices[0].tolist()
                preds.append([self.idx2item.get(i, -1) for i in top])
        return preds

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.model.state_dict(), "item2idx": self.item2idx}, path)

    @classmethod
    def load(cls, path: Path) -> "RNNRecModel":
        data = torch.load(path, map_location="cpu")
        model = GRURecModel(len(data["item2idx"]) + 1)
        model.load_state_dict(data["state_dict"])
        return cls(model=model, item2idx=data["item2idx"])
