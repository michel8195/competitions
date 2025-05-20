import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def compute_item_bigrams(df: pd.DataFrame, session_col: str, item_col: str) -> pd.DataFrame:
    """Return DataFrame with columns [prev_item, next_item, count]."""
    df = df[[session_col, item_col]].copy()
    df['next_item'] = df.groupby(session_col)[item_col].shift(-1)
    bigrams = df.dropna(subset=['next_item']).groupby([item_col, 'next_item']).size().reset_index(name='count')
    bigrams.rename(columns={item_col: 'prev_item'}, inplace=True)
    return bigrams


def compute_item_popularity(df: pd.DataFrame, item_col: str) -> pd.Series:
    return df[item_col].value_counts()


def train(train_path: Path, session_col: str, item_col: str, bigram_path: Path, popularity_path: Path) -> None:
    df = pd.read_csv(train_path)
    bigrams = compute_item_bigrams(df, session_col, item_col)
    bigrams.to_csv(bigram_path, index=False)
    popularity = compute_item_popularity(df, item_col)
    popularity.to_csv(popularity_path, header=False)
    print(f"Saved bigram counts to {bigram_path}")
    print(f"Saved popularity counts to {popularity_path}")


def load_bigrams(bigram_path: Path) -> pd.DataFrame:
    return pd.read_csv(bigram_path)


def load_popularity(popularity_path: Path) -> pd.Series:
    pop = pd.read_csv(popularity_path, header=None, names=['item', 'count'])
    return pop.set_index('item')['count']


def predict_sessions(df_sessions: pd.DataFrame, bigrams: pd.DataFrame, popularity: pd.Series, session_col: str, item_col: str, top_k: int) -> List[List[int]]:
    bigram_map: Dict[int, pd.Series] = {
        prev: grp.set_index('next_item')['count'] for prev, grp in bigrams.groupby('prev_item')
    }
    popular_items = popularity.sort_values(ascending=False).index.tolist()
    predictions = []
    for _, session in df_sessions.groupby(session_col):
        last_item = session[item_col].iloc[-1]
        if last_item in bigram_map:
            next_counts = bigram_map[last_item].sort_values(ascending=False)
            recs = next_counts.head(top_k).index.tolist()
        else:
            recs = []
        if len(recs) < top_k:
            for item in popular_items:
                if item not in recs:
                    recs.append(item)
                if len(recs) == top_k:
                    break
        predictions.append(recs)
    return predictions


def predict(bigram_path: Path, popularity_path: Path, sessions_path: Path, output_path: Path, session_col: str, item_col: str, top_k: int) -> None:
    bigrams = load_bigrams(bigram_path)
    popularity = load_popularity(popularity_path)
    df = pd.read_csv(sessions_path)
    predictions = predict_sessions(df, bigrams, popularity, session_col, item_col, top_k)
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Item bigram baseline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_path", type=Path, help="Training CSV file")
    train_parser.add_argument("bigram_path", type=Path, help="Path to save bigram counts")
    train_parser.add_argument("popularity_path", type=Path, help="Path to save popularity counts")
    train_parser.add_argument("--session-col", default="session_id", help="Column with session id")
    train_parser.add_argument("--item-col", default="product_id", help="Column with item id")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("bigram_path", type=Path, help="Bigram counts file from training")
    predict_parser.add_argument("popularity_path", type=Path, help="Popularity file from training")
    predict_parser.add_argument("sessions_path", type=Path, help="CSV file with test sessions")
    predict_parser.add_argument("output_path", type=Path, help="Path to save predictions JSON")
    predict_parser.add_argument("--session-col", default="session_id")
    predict_parser.add_argument("--item-col", default="product_id")
    predict_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        train(args.train_path, args.session_col, args.item_col, args.bigram_path, args.popularity_path)
    elif args.command == "predict":
        predict(args.bigram_path, args.popularity_path, args.sessions_path, args.output_path,
                args.session_col, args.item_col, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
