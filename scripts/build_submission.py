"""Assemble the final submission zip in the structure the challenge specifies.

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   +-- candidate_pairs.tsv
    |-- code/business_entity_resolution/
    |   |-- src/
    |   |-- README.md
    |   +-- requirements.txt
    +-- Documentation_template.md

Run after scripts/submit.py has produced both TSVs:

    python scripts/build_submission.py --team myteam

Refuses to build if either output file is missing or does not have exactly one row per test
Source-1 entity, because a submission short of rows is rejected outright rather than scored
down, and that is not something to discover after uploading.
"""

import argparse
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.config import OUTPUT, REPO

PACKAGING = os.path.join(REPO, "packaging")
STAGE = os.path.join(REPO, "submission")
CODE_DIR = "code/business_entity_resolution"


def count_rows(path):
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        return sum(1 for line in fh if line.strip("\n"))


def expected_entities():
    path = os.path.join(REPO, "student_resource", "dataset", "test", "test_source1.tsv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        return sum(1 for line in fh if line.strip("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name, used for the zip filename")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build even if the outputs are short of a full test set")
    args = ap.parse_args()

    matching = os.path.join(OUTPUT, "matching_results.tsv")
    candidate = os.path.join(OUTPUT, "candidate_pairs.tsv")
    for p in (matching, candidate):
        if not os.path.exists(p):
            sys.exit(f"missing {p} - run scripts/submit.py first")

    n_m, n_c = count_rows(matching), count_rows(candidate)
    need = expected_entities()
    print(f"matching_results.tsv : {n_m:,} rows")
    print(f"candidate_pairs.tsv  : {n_c:,} rows")
    if need:
        print(f"test Source-1 entities: {need:,}")
    if n_m != n_c:
        sys.exit(f"row count mismatch: {n_m:,} vs {n_c:,}")
    if need and n_m != need and not args.allow_partial:
        sys.exit(f"outputs cover {n_m:,} of {need:,} entities. A submission missing rows is "
                 f"rejected, not scored down. Finish the run, or pass --allow-partial.")

    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "output"))
    code_root = os.path.join(STAGE, *CODE_DIR.split("/"))
    os.makedirs(os.path.join(code_root, "src"))

    shutil.copy2(matching, os.path.join(STAGE, "output", "matching_results.tsv"))
    shutil.copy2(candidate, os.path.join(STAGE, "output", "candidate_pairs.tsv"))

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pkl", ".ipynb_checkpoints")
    shutil.copytree(os.path.join(REPO, "src", "ber"),
                    os.path.join(code_root, "src", "ber"), ignore=ignore)
    shutil.copytree(os.path.join(REPO, "scripts"),
                    os.path.join(code_root, "src", "scripts"), ignore=ignore)

    shutil.copy2(os.path.join(PACKAGING, "code_README.md"),
                 os.path.join(code_root, "README.md"))
    shutil.copy2(os.path.join(PACKAGING, "code_requirements.txt"),
                 os.path.join(code_root, "requirements.txt"))
    shutil.copy2(os.path.join(PACKAGING, "Documentation_template.md"),
                 os.path.join(STAGE, "Documentation_template.md"))

    zip_path = os.path.join(REPO, f"{args.team}_submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(STAGE):
            for name in sorted(files):
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, STAGE).replace(os.sep, "/"))

    print(f"\nwrote {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    print(f"{len(names)} entries; structure:")
    for n in sorted(names):
        if n.count("/") <= 2 or n.endswith((".md", ".txt", ".tsv")):
            print(f"  {n}")

    required = ["output/matching_results.tsv", "output/candidate_pairs.tsv",
                f"{CODE_DIR}/README.md", f"{CODE_DIR}/requirements.txt",
                "Documentation_template.md"]
    missing = [r for r in required if r not in names]
    if missing:
        sys.exit(f"ERROR: zip is missing required entries: {missing}")
    if not any(n.startswith(f"{CODE_DIR}/src/") for n in names):
        sys.exit(f"ERROR: no source files under {CODE_DIR}/src/")
    print("\nstructure matches the required layout")


if __name__ == "__main__":
    main()
