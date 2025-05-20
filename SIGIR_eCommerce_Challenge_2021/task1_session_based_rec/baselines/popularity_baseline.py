import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from models import PopularityModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Popularity baseline")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_path", type=Path, help="Training CSV file")
    train_parser.add_argument("output_path", type=Path, help="Path to save model")
    train_parser.add_argument("--item-col", default="product_id", help="Column with item id")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("model_path", type=Path, help="Trained model file")
    predict_parser.add_argument("output_path", type=Path, help="Path to save predictions JSON")
    predict_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        model = PopularityModel()
        model.train(args.train_path, item_col=args.item_col)
        model.save(args.output_path)
    elif args.command == "predict":
        model = PopularityModel.load(args.model_path)
        # predictions are independent of sessions
        predictions = model.predict_sessions([[]], args.top_k)
        with open(args.output_path, "w") as f:
            json.dump(predictions, f)
        print(f"Saved predictions to {args.output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
