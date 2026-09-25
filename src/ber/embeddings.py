"""Lookup side of the multilingual name encoder.

Deliberately numpy-only: the encoder runs as a separate precompute step in the CUDA venv on
D:, writes a memmap, and the hot pipeline just reads it. That keeps torch out of the
scoring loop and keeps the pipeline itself dependency-free.

**We embed only what the encoder can actually help with.** S1 names are always Latin; the
gap is the 7.3% of true pairs whose S2/S3 name is in one of ten Indic scripts, where token
and character similarity are both structurally zero because the two sides share no alphabet.
So the encoder covers Source-1 names plus *only* the non-Latin Source-2/3 names - about
2.4M texts instead of 11.7M. Everything else is already handled better, and more cheaply, by
the string features.
"""

import json
import os

import numpy as np


class NameEmbeddings:
    """Memory-mapped name vectors with an entity-id index.

    Vectors are stored L2-normalised and float16, so similarity is a plain dot product and
    2.4M x 384 costs ~1.8 GB on disk.
    """

    def __init__(self, directory, name="names"):
        self.directory = directory
        meta_path = os.path.join(directory, f"{name}.meta.json")
        with open(meta_path, encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.dim = int(self.meta["dim"])
        self.count = int(self.meta["count"])
        self.model = self.meta.get("model", "?")
        self.vectors = np.memmap(
            os.path.join(directory, f"{name}.f16"),
            dtype=np.float16, mode="r", shape=(self.count, self.dim),
        )
        with open(os.path.join(directory, f"{name}.ids"), encoding="utf-8") as fh:
            self.row_of = {line.rstrip("\n"): i for i, line in enumerate(fh)}

    def __contains__(self, entity_id):
        return entity_id in self.row_of

    def get(self, entity_id):
        """Unit vector for an entity, or None if it was not embedded."""
        row = self.row_of.get(entity_id)
        return None if row is None else self.vectors[row]

    def similarity(self, id_a, id_b):
        """Cosine similarity in [0, 1], or None when either side has no vector.

        Vectors are pre-normalised so this is a dot product. The raw cosine is rescaled from
        [-1, 1] to [0, 1] to match the range of every other feature in the scorer.
        """
        ra = self.row_of.get(id_a)
        if ra is None:
            return None
        rb = self.row_of.get(id_b)
        if rb is None:
            return None
        dot = float(np.dot(self.vectors[ra].astype(np.float32),
                           self.vectors[rb].astype(np.float32)))
        if dot < -1.0:
            dot = -1.0
        elif dot > 1.0:
            dot = 1.0
        return 0.5 * (dot + 1.0)


def load(directory, name="names"):
    """Return NameEmbeddings, or None when the precompute step has not been run."""
    try:
        return NameEmbeddings(directory, name)
    except (FileNotFoundError, KeyError, ValueError):
        return None
