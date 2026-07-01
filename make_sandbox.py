"""
It creates a sandbox/ folder containing:
  - sandbox/candidates_sample.jsonl   (50 real candidates)
  - sandbox/artifacts/*.npy + candidate_ids.json  (matching rows only)

This is what Colab will run against. No manual candidate picking needed.
"""

import json
import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data",
                     help="Folder containing candidates.jsonl (defaults to ./data)")
args = parser.parse_args()

DATA_DIR = args.data_dir
ARTIFACT_DIR = "artifacts"
OUT_DIR = "sandbox"
OUT_ARTIFACT_DIR = os.path.join(OUT_DIR, "artifacts")
N_SAMPLE = 100

os.makedirs(OUT_ARTIFACT_DIR, exist_ok=True)

# --- Load full candidate file (auto-detect JSON array vs true JSONL) ---
cand_path = os.path.join(DATA_DIR, "candidates.jsonl")
with open(cand_path, "r", encoding="utf-8") as f:
    first_char = f.read(1)
    f.seek(0)
    if first_char == "[":
        candidates = json.load(f)
    else:
        candidates = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(candidates)} candidates from {cand_path}")

# --- Take first 50 ---
sample = candidates[:N_SAMPLE]
sample_ids = [c["candidate_id"] for c in sample]
print(f"Sampled {len(sample)} candidates: {sample_ids[0]} ... {sample_ids[-1]}")

# --- Write sandbox candidates file (JSON array, same format rank.py already handles) ---
with open(os.path.join(OUT_DIR, "candidates_sample.jsonl"), "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

# --- Load full candidate_ids.json to find index positions for our sample ---
with open(os.path.join(ARTIFACT_DIR, "candidate_ids.json"), "r", encoding="utf-8") as f:
    all_ids = json.load(f)

id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
sample_indices = [id_to_idx[cid] for cid in sample_ids]

# --- Slice each .npy artifact to just those rows, preserving order ---
for fname in ["embeddings.npy", "career_features.npy", "behavioral_features.npy", "honeypot_flags.npy"]:
    full = np.load(os.path.join(ARTIFACT_DIR, fname))
    sliced = full[sample_indices]
    np.save(os.path.join(OUT_ARTIFACT_DIR, fname), sliced)
    print(f"  {fname}: {full.shape} -> {sliced.shape}")

# --- Write matching candidate_ids.json for the sample ---
with open(os.path.join(OUT_ARTIFACT_DIR, "candidate_ids.json"), "w", encoding="utf-8") as f:
    json.dump(sample_ids, f)

# --- Copy JD embedding as-is (same for every candidate set) ---
import shutil
shutil.copy(
    os.path.join(ARTIFACT_DIR, "jd_embedding.npy"),
    os.path.join(OUT_ARTIFACT_DIR, "jd_embedding.npy"),
)

print("\nDone. Sandbox folder created at:", OUT_DIR)
