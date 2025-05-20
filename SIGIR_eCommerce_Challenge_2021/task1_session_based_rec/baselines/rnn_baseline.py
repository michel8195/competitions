import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from models import RNNRecModel


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU-based session model")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_path", type=Path, help="Training CSV file")
    train_parser.add_argument("model_path", type=Path, help="Path to save model")
    train_parser.add_argument("--session-col", default="session_id")
    train_parser.add_argument("--item-col", default="product_id")
    train_parser.add_argument("--epochs", type=int, default=5)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("model_path", type=Path, help="Trained model file")
    predict_parser.add_argument("interactions_path", type=Path, help="CSV with session interactions")
    predict_parser.add_argument("output_path", type=Path, help="Path to save predictions JSON")
    predict_parser.add_argument("--session-col", default="session_id")
    predict_parser.add_argument("--item-col", default="product_id")
    predict_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        model = RNNRecModel()
        model.train(
            args.train_path,
            session_col=args.session_col,
            item_col=args.item_col,
            epochs=args.epochs,
        )
        model.save(args.model_path)
    elif args.command == "predict":
        model = RNNRecModel.load(args.model_path)
        df = pd.read_csv(args.interactions_path)
        sessions = df.groupby(args.session_col)[args.item_col].apply(list).tolist()
        predictions = model.predict_sessions(sessions, top_k=args.top_k)
        with open(args.output_path, "w") as f:
            json.dump(predictions, f)
        print(f"Saved predictions to {args.output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
