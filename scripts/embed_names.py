"""Precompute multilingual name embeddings on the GPU.

Runs in the CUDA venv on D: (torch + sentence-transformers) and writes a memmap that the
main pipeline reads with numpy alone, so torch never enters the scoring loop.

    D:\\ml_challenge\\venv\\Scripts\\python.exe scripts/embed_names.py --split test

**What gets embedded, and why so little.** Source-1 names are always Latin. The gap the
encoder exists to close is the 7.3% of true pairs whose Source-2/3 name is in one of ten
Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia,
Gurmukhi) or Arabic - cases where token and character similarity are *structurally* zero
because the two sides share no alphabet. Everything else the string features already handle
better and far more cheaply.

So we embed all Source-1 names plus only the non-Latin Source-2/3 names: ~2.4M texts rather
than 11.7M. ``--all-targets`` overrides this if we ever want to test the encoder as a general
similarity instead of a targeted patch.

Model must be MIT or Apache-2.0 and at most 8B parameters per the challenge rules. LaBSE
(Apache-2.0, 471M) is the default because it was trained on translation pairs across 109
languages, which is exactly the romanised-vs-native-script problem here.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ber.config import EMBEDDINGS, MODELS, TEST, TRAIN, ensure_dirs
from ber.normalize import is_latin

MODEL_CHOICES = {
    # name: (hf id, licence, params)
    "labse": ("sentence-transformers/LaBSE", "Apache-2.0", "471M"),
    "e5-small": ("intfloat/multilingual-e5-small", "MIT", "118M"),
    "e5-base": ("intfloat/multilingual-e5-base", "MIT", "278M"),
    "bge-m3": ("BAAI/bge-m3", "MIT", "568M"),
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def collect(split, all_targets):
    """Yield (entity_id, name) for every record that needs a vector."""
    base = TRAIN if split == "train" else TEST
    prefix = f"{split}_source"
    # Source 1: always embedded, it is the reference side of every comparison.
    for source, only_non_latin in ((1, False), (2, not all_targets), (3, not all_targets)):
        path = os.path.join(base, f"{prefix}{source}.tsv")
        with open(path, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                if only_non_latin and is_latin(parts[1]):
                    continue
                yield parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--model", choices=sorted(MODEL_CHOICES), default="labse")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--all-targets", action="store_true",
                    help="embed every S2/S3 name, not just non-Latin ones")
    ap.add_argument("--limit", type=int, default=0, help="stop after N texts (smoke test)")
    args = ap.parse_args()

    ensure_dirs()
    hf_id, licence, params = MODEL_CHOICES[args.model]

    # Keep the HF cache on D: as well; weights are hundreds of MB.
    os.environ.setdefault("HF_HOME", MODELS)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODELS)

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        log("WARNING: CUDA unavailable, this will be extremely slow on CPU")
    else:
        log(f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.mem_get_info()[1] / 1e9:.1f} GB)")

    log(f"collecting texts for split={args.split} ...")
    ids, texts = [], []
    for eid, name in collect(args.split, args.all_targets):
        ids.append(eid)
        texts.append(name)
        if args.limit and len(ids) >= args.limit:
            break
    log(f"{len(ids):,} texts to embed "
        f"({'all targets' if args.all_targets else 'S1 + non-Latin S2/S3 only'})")

    log(f"loading {hf_id} ({licence}, {params}) ...")
    model = SentenceTransformer(hf_id, device="cuda" if torch.cuda.is_available() else "cpu",
                                cache_folder=MODELS)
    model = model.half()  # fp16: names are short, and 8 GB of VRAM is the budget
    dim = model.get_sentence_embedding_dimension()
    log(f"embedding dimension {dim}")

    out = os.path.join(EMBEDDINGS, f"{args.split}_names")
    vec_path = out + ".f16"
    mm = np.memmap(vec_path, dtype=np.float16, mode="w+", shape=(len(ids), dim))

    t0 = time.time()
    step = args.batch_size * 40
    for start in range(0, len(texts), step):
        chunk = texts[start:start + step]
        emb = model.encode(chunk, batch_size=args.batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        mm[start:start + len(chunk)] = emb.astype(np.float16)
        done = start + len(chunk)
        rate = done / (time.time() - t0)
        log(f"  {done:,}/{len(ids):,}  {rate:,.0f} texts/s  "
            f"eta {(len(ids) - done) / max(rate, 1) / 60:.1f} min")
    mm.flush()
    del mm

    with open(out + ".ids", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(ids))
    with open(out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump({"model": hf_id, "licence": licence, "params": params, "dim": dim,
                   "count": len(ids), "split": args.split, "fp16": True,
                   "normalised": True, "all_targets": bool(args.all_targets)}, fh, indent=2)

    size_gb = os.path.getsize(vec_path) / 1e9
    log(f"wrote {out}.f16 ({size_gb:.2f} GB), .ids, .meta.json in "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
