import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def compute_item_popularity(df: pd.DataFrame, item_col: str) -> pd.Series:
    """Return item popularity sorted descending."""
    return df[item_col].value_counts()


def train(train_path: Path, item_col: str, output_path: Path) -> None:
    df = pd.read_csv(train_path)
    popularity = compute_item_popularity(df, item_col)
    popularity.to_csv(output_path, header=False)
    print(f"Saved popularity counts to {output_path}")


def load_popularity(popularity_path: Path) -> pd.Series:
    pop = pd.read_csv(popularity_path, header=None, names=['item', 'count'])
    return pop.set_index('item')['count']


def predict_sessions(popularity: pd.Series, top_k: int) -> List[List[int]]:
    top_items = popularity.sort_values(ascending=False).head(top_k).index.tolist()
    return [top_items]


def predict(popularity_path: Path, output_path: Path, top_k: int) -> None:
    popularity = load_popularity(popularity_path)
    predictions = predict_sessions(popularity, top_k)
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Popularity baseline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_path", type=Path, help="Training CSV file")
    train_parser.add_argument("output_path", type=Path, help="Path to save popularity counts")
    train_parser.add_argument("--item-col", default="product_id", help="Column with item id")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("popularity_path", type=Path, help="Popularity file from training")
    predict_parser.add_argument("output_path", type=Path, help="Path to save predictions JSON")
    predict_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        train(args.train_path, args.item_col, args.output_path)
    elif args.command == "predict":
        predict(args.popularity_path, args.output_path, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
