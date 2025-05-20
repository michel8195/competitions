from __future__ import annotations

from typing import Dict, List
from pathlib import Path

import torch
from torch import nn
import pytorch_lightning as pl


class GRURecModel(pl.LightningModule):
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))

    def training_step(self, batch, batch_idx):
        seq, target = batch
        logits = self(seq)
        loss = self.criterion(logits, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict_sessions(self, sessions: List[List[int]], item2idx: Dict[int, int], idx2item: Dict[int, int], top_k: int = 20) -> List[List[int]]:
        device = self.device
        preds: List[List[int]] = []
        self.eval()
        with torch.no_grad():
            for s in sessions:
                idxs = [item2idx.get(i, 0) for i in s][-20:]
                seq = torch.tensor([idxs], dtype=torch.long, device=device)
                logits = self(seq)
                top = logits.topk(top_k, dim=1).indices[0].tolist()
                preds.append([idx2item.get(i, -1) for i in top])
        return preds

    def save(self, path: Path):
        torch.save({'state_dict': self.state_dict(), 'hyper_parameters': self.hparams}, path)

    @classmethod
    def load(cls, path: Path) -> "GRURecModel":
        data = torch.load(path, map_location='cpu')
        model = cls(**data['hyper_parameters'])
        model.load_state_dict(data['state_dict'])
        return model
