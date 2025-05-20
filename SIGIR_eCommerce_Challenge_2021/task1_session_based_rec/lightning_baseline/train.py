import argparse
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import SessionDataModule
from model import GRURecModel


def main():
    parser = argparse.ArgumentParser(description="Train GRU session model with PyTorch Lightning")
    parser.add_argument("train_path", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--item-col", default="product_id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data = SessionDataModule(args.train_path, args.session_col, args.item_col, batch_size=args.batch_size)
    data.setup()
    model = GRURecModel(num_items=len(data.item2idx) + 1)
    checkpoint = ModelCheckpoint(dirpath=str(args.model_path.parent), filename=args.model_path.stem, save_last=True)
    trainer = Trainer(max_epochs=args.epochs, callbacks=[checkpoint])
    trainer.fit(model, datamodule=data)
    model.save(args.model_path)


if __name__ == "__main__":
    main()
