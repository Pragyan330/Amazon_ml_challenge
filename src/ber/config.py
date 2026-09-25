"""Paths and storage layout.

Large artefacts live on D: — the C: system drive has ~20 GB free, which is not enough for
model weights plus embeddings for 11.7M records. Override with BER_WORK if needed.
"""

import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Everything bulky goes here: cached statistics, model weights, embedding memmaps.
WORK = os.environ.get("BER_WORK", r"D:\ml_challenge")

ARTIFACTS = os.path.join(WORK, "artifacts")
MODELS = os.path.join(WORK, "models")
EMBEDDINGS = os.path.join(WORK, "embeddings")

# Small enough to keep beside the code.
OUTPUT = os.path.join(REPO, "output")

DATA = os.path.join(REPO, "student_resource", "dataset")
TRAIN = os.path.join(DATA, "train")
TEST = os.path.join(DATA, "test")


def train_paths():
    return {
        "s1": os.path.join(TRAIN, "train_source1.tsv"),
        "s23": [os.path.join(TRAIN, "train_source2.tsv"),
                os.path.join(TRAIN, "train_source3.tsv")],
        "gt": os.path.join(TRAIN, "train_ground_truth.tsv"),
    }


def test_paths():
    return {
        "s1": os.path.join(TEST, "test_source1.tsv"),
        "s23": [os.path.join(TEST, "test_source2.tsv"),
                os.path.join(TEST, "test_source3.tsv")],
    }


def ensure_dirs():
    for d in (ARTIFACTS, MODELS, EMBEDDINGS, OUTPUT):
        os.makedirs(d, exist_ok=True)
    return {"work": WORK, "artifacts": ARTIFACTS, "models": MODELS,
            "embeddings": EMBEDDINGS, "output": OUTPUT}
