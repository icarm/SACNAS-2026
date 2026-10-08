"""Regenerate the committed checkpoints. Run from the repo root:

    python tools/make_checkpoints.py

Trains the main model (one-hot digits, width 512, plain SGD, seed 0) and saves
the network at epochs 0, 30 and 400 plus the per-epoch history.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import mlmath

data = mlmath.make_data()
model, history, saved = mlmath.train(mlmath.encode_onehot, data, save_at=(0, 30, 400), live=False)
for epoch, snapshot in saved.items():
    mlmath.save_model(snapshot, f"checkpoints/mod6_epoch{epoch:03d}.pt")
mlmath.save_history(history, "checkpoints/mod6_history.csv")
mlmath.summarize(history)
