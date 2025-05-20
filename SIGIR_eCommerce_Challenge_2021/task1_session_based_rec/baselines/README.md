# Baseline Solutions for Task 1

This folder contains simple reference implementations for the session-based recommendation task.
The goal is to provide lightweight starting points before moving to the complex Transformer-based models used in our final solution.

## Popularity Baseline
The `popularity_baseline.py` script computes the most popular items in the training data and recommends the top items for every session.
It requires the input interactions in CSV format with an `product_id` column.

### Train
```bash
python popularity_baseline.py train train_interactions.csv popularity_model.pkl
```

### Predict
```bash
python popularity_baseline.py predict popularity_model.pkl predictions.json --top-k 20
```
The prediction file will contain a JSON array with the same list of top items for every session.

## GRU PyTorch Baseline
The `rnn_baseline.py` script provides a minimal GRU-based model implemented with PyTorch. It shares the same command line interface as the popularity baseline and saves models to disk using `torch.save`.

### Train
```bash
python rnn_baseline.py train train_interactions.csv rnn_model.pth --epochs 5
```

### Predict
```bash
python rnn_baseline.py predict rnn_model.pth interactions.csv predictions.json --top-k 20
```

These baselines can be extended with more sophisticated logic (e.g. item bigrams or session-aware heuristics) as needed.
