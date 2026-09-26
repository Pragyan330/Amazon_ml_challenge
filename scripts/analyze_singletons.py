import pickle
import os

path = r"D:\ml_challenge\artifacts\val_scored.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

scored = data["scored"]
truth = data["truth"]
best_threshold = data["best_threshold"] # 0.72

# We want Source 1 entities that are true singletons (truth[eid] is empty) 
# but got matched (some candidate scored >= best_threshold)
false_singletons = []

for eid, true_matches in truth.items():
    if len(true_matches) == 0:
        # It's a singleton
        candidates = scored.get(eid, [])
        predicted = [c for c in candidates if c[0] >= best_threshold]
        if len(predicted) > 0:
            false_singletons.append((eid, predicted))

print(f"Total true singletons in validation: {sum(1 for v in truth.values() if len(v) == 0)}")
print(f"Total false singletons (matched at >= {best_threshold}): {len(false_singletons)}")

# Let's print 15-20 examples
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.pipeline import read_source

s1_path = "student_resource/dataset/train/train_source1.tsv"
s2_path = "student_resource/dataset/train/train_source2.tsv"
s3_path = "student_resource/dataset/train/train_source3.tsv"

s1_dict = {}
with open(s1_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip('\n').split('\t')
        if len(parts) >= 4:
            s1_dict[parts[0]] = parts

s23_dict = {}
for p in [s2_path, s3_path]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s23_dict[parts[0]] = parts

print("\n--- Examples of singletons falsely matched ---")
for eid, preds in false_singletons[:20]:
    print(f"\nS1 Entity: {eid} -> {s1_dict.get(eid)}")
    for score, cid in preds:
        print(f"  Matched (score {score:.4f}): {cid} -> {s23_dict.get(cid)}")
