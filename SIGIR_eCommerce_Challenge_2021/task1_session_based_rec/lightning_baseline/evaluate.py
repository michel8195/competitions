import argparse
import json
from pathlib import Path
import pandas as pd

from datasets import SessionDataModule
from model import GRURecModel


def recall_at_k(predictions, targets, k: int = 20) -> float:
    assert len(predictions) == len(targets)
    hits = 0
    for pred, target in zip(predictions, targets):
        if target in pred[:k]:
            hits += 1
    return hits / len(predictions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRU model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("interactions_path", type=Path)
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--item-col", default="product_id")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = GRURecModel.load(args.model_path)
    dm = SessionDataModule(args.interactions_path, args.session_col, args.item_col)
    dm.setup()
    sessions = dm.dataset.inputs
    targets = dm.dataset.targets
    predictions = model.predict_sessions(sessions, dm.item2idx, dm.idx2item, top_k=args.top_k)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f)
    rec = recall_at_k(predictions, targets, args.top_k)
    print(f"Recall@{args.top_k}: {rec:.4f}")


if __name__ == "__main__":
    main()
