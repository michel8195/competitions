# Baseline Solutions for Task 1

This folder contains simple reference implementations for the session-based recommendation task.
The goal is to provide lightweight starting points before moving to the complex Transformer-based models used in our final solution.

## Popularity Baseline
The `popularity_baseline.py` script computes the most popular items in the training data and recommends the top items for every session.
It requires the input interactions in CSV format with a `product_id` column.

### Train
```bash
python popularity_baseline.py train train_interactions.csv popularity.csv
```

### Predict
```bash
python popularity_baseline.py predict popularity.csv predictions.json --top-k 20
```
The prediction file will contain a JSON array with the same list of top items for every session.

## Item Bigram Baseline
The `item_bigram_baseline.py` script extends the popularity approach by also
considering the last item of each session. It computes item-to-item bigram
frequencies from the training data and recommends the most likely next items for
each session. When a transition is unknown, it falls back to the overall item
popularity.

### Train
```bash
python item_bigram_baseline.py train train_interactions.csv bigrams.csv popularity.csv \
    --session-col session_id --item-col product_id
```

### Predict
```bash
python item_bigram_baseline.py predict bigrams.csv popularity.csv test_interactions.csv \
    predictions.json --top-k 20 --session-col session_id --item-col product_id
```
The predictions file will contain a JSON array with one list of recommended
items per session ordered by score.
