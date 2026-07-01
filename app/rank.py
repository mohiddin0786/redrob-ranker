"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

Architecture: Two-phase offline/online system.
  - Offline (precompute.py): embeddings + feature engineering → artifacts/
  - Online (this file):      load artifacts → score → top-100 CSV

Scoring weights:
  40% semantic skill match (cosine similarity of embeddings)
  35% career signal        (weighted combo of career_features)
  25% behavioral multiplier (weighted combo of behavioral_features)

Compute profile: CPU-only, ~16 GB RAM, <5 min wall-clock for 100K candidates.
"""

import os
import sys
import gzip
import json
import argparse
import csv
from datetime import date

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR      = os.environ.get("DATA_DIR",      "data")
ARTIFACT_DIR  = os.environ.get("ARTIFACT_DIR",  "artifacts")
DEFAULT_OUT   = os.environ.get("OUTPUT_PATH",   "submission/team_vecForge.csv")

CANDIDATES_FILE     = os.path.join(DATA_DIR,     "candidates.jsonl")
EMBEDDINGS_FILE     = os.path.join(ARTIFACT_DIR, "embeddings.npy")
CAREER_FEAT_FILE    = os.path.join(ARTIFACT_DIR, "career_features.npy")
BEHAVIORAL_FEAT_FILE= os.path.join(ARTIFACT_DIR, "behavioral_features.npy")
CANDIDATE_IDS_FILE  = os.path.join(ARTIFACT_DIR, "candidate_ids.json")
HONEYPOT_MASK_FILE  = os.path.join(ARTIFACT_DIR, "honeypot_flags.npy")

# ── Scoring weights ────────────────────────────────────────────────────────────

W_SEMANTIC = 0.40   # cosine similarity of candidate embedding vs JD query
W_CAREER   = 0.35   # career feature score
W_BEHAVIORAL = 0.25 # behavioral multiplier

# Career feature sub-weights (must sum to 1.0)
# Indices match extract_career_features() output order:
# 0=yoe, 1=has_product, 2=consulting_penalty, 3=research_penalty,
# 4=longest_tenure, 5=title_score, 6=prod_signal, 7=ai_prod_signal,
# 8=skill_depth, 9=location_match
CW = np.array([
    0.08,  # 0 yoe              — tent function
    0.15,  # 1 has_product      — product vs services
    0.10,  # 2 consulting_pen   — multiplicative
    0.05,  # 3 research_pen     — multiplicative
    0.10,  # 4 domain_fit       — CV/speech penalty (was longest_tenure)
    0.22,  # 5 title_score      — anti-keyword-stuffer (trim slightly)
    0.10,  # 6 prod_signal      — production keywords
    0.08,  # 7 ai_prod_signal   — AI prod keywords
    0.05,  # 8 skill_depth      — duration-backed skills
    0.07,  # 9 location_match   — preferred cities
], dtype=np.float32)
assert abs(CW.sum() - 1.0) < 1e-5, "Career weights must sum to 1.0"

# Behavioral feature sub-weights (must sum to 1.0)
# 0=recency, 1=open_to_work, 2=response_rate, 3=interview_completion,
# 4=profile_completeness, 5=github_activity, 6=saved_by_recruiters,
# 7=notice_score, 8=apps_score, 9=verification
BW = np.array([
    0.25,  # 0 recency                 — inactive candidates not hirable
    0.15,  # 1 open_to_work            — explicit intent signal
    0.15,  # 2 recruiter_response_rate — will they reply?
    0.10,  # 3 interview_completion    — will they show up?
    0.05,  # 4 profile_completeness    — data quality
    0.10,  # 5 github_activity         — external validation (JD wants this)
    0.05,  # 6 saved_by_recruiters     — market validation
    0.10,  # 7 notice_score            — JD prefers sub-30d
    0.03,  # 8 apps_score              — active job search signal
    0.02,  # 9 verification            — profile authenticity
], dtype=np.float32)
assert abs(BW.sum() - 1.0) < 1e-5, "Behavioral weights must sum to 1.0"


# ── JD embedding ──────────────────────────────────────────────────────────────
def load_jd_embedding(artifact_dir: str) -> np.ndarray:
    """Load precomputed JD embedding artifact (computed offline in Colab)."""
    path = os.path.join(artifact_dir, "jd_embedding.npy")
    vec = np.load(path)
    return vec.astype(np.float32)  # shape (384,)


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(
    jd_vec: np.ndarray,
    embeddings: np.ndarray,
    career_feat: np.ndarray,
    behavioral_feat: np.ndarray,
    Invalid_mask: np.ndarray,
) -> np.ndarray:
    """
    Returns array of shape (N,) with final scores.
    Honeypots / invalid candidates get score = -1.0 (guaranteed out of top 100).

    Steps:
      1. Cosine similarity: single BLAS matmul (~50ms for 100K x 384)
      2. Career score:    dot(career_feat, CW)
      3. Behavioral score: dot(behavioral_feat, BW)
      4. Final:           W_SEMANTIC*sem + W_CAREER*car + W_BEHAVIORAL*beh
      5. Zero-out honeypots
    """
    N = embeddings.shape[0]

    # 1. Semantic similarity (embeddings already L2-normalized from precompute)
    # jd_vec shape: (384,) → matmul gives (N,)
    sem_scores = embeddings @ jd_vec  # cosine sim, range [-1, 1] → clamp to [0,1]
    sem_scores = np.clip(sem_scores, 0.0, 1.0)

    # Stretch to use full [0,1] range based on rank position
    # Rank 1 → 1.0, Rank 100000 → ~0.0, everyone else in between

    # 2. Career score: weighted sum of career features
    car_scores = career_feat @ CW  # shape (N,)
    # consulting_penalty (col 2) and research_penalty (col 3) are already
    # baked in as features (0.3 or 1.0), so the weighted sum naturally
    # suppresses all-consulting / all-research candidates.
    car_scores = car_scores * career_feat[:, 2] * career_feat[:, 3]* career_feat[:, 4]

    # 3. Behavioral score: weighted sum of behavioral features
    beh_scores = behavioral_feat @ BW  # shape (N,)

    # 4. Combine
    final = (W_SEMANTIC * sem_scores +
             W_CAREER   * car_scores +
             W_BEHAVIORAL * beh_scores)

    # 5. Hard-zero honeypots (they'll never enter top 100)
    final[Invalid_mask] = -1.0

    return final.astype(np.float64)


# ── Reasoning strings ─────────────────────────────────────────────────────────
def build_reasoning(c: dict, score: float, rank: int) -> str:
    profile  = c.get("profile", {})
    signals  = c.get("redrob_signals", {})
    career   = c.get("career_history", [])
    skills   = c.get("skills", [])

    title    = profile.get("current_title", "")
    yoe      = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    notice   = signals.get("notice_period_days", 90)
    github   = signals.get("github_activity_score", -1)
    response = signals.get("recruiter_response_rate", 0.0)
    open_w   = signals.get("open_to_work_flag", False)
    relocate = signals.get("willing_to_relocate", False)

    # IR/retrieval skills specifically — not just any expert skill
    IR_SKILLS = {'faiss','pinecone','weaviate','qdrant','milvus','opensearch',
                 'elasticsearch','pgvector','bm25','vector search','semantic search',
                 'information retrieval','learning to rank','recommendation systems',
                 'embeddings','sentence transformers','dense retrieval','hybrid search',
                 'rag','reranking','lora','qlora','peft'}
    
    expert_ir = [s['name'] for s in skills 
                 if s.get('proficiency') in ('expert','advanced') 
                 and s['name'].lower() in IR_SKILLS][:3]
    
    # Product companies in career
    PRODUCT_INDS = {'internet','fintech','food delivery','e-commerce','edtech',
                    'saas','ai/ml','media','adtech','healthtech'}
    product_cos = [r['company'] for r in career 
                   if r.get('industry','').lower() in PRODUCT_INDS][:2]
    
    # Recent company for context
    recent = career[0] if career else {}
    recent_co = recent.get('company','')
    recent_dur = recent.get('duration_months', 0)
    
    # Build sentence 1: specific career fact
    if expert_ir:
        skill_str = ', '.join(expert_ir)
        s1 = f"{yoe:.0f}yr {title} ({location}) with hands-on {skill_str} experience"
        if product_cos:
            s1 += f" across {', '.join(product_cos)}"
    else:
        s1 = f"{yoe:.0f}yr {title} at {recent_co} ({location})"
    s1 += "."

    # Build sentence 2: fit assessment with honest concerns
    notes = []
    
    # Positive signals
    if notice <= 30:
        notes.append(f"available in {notice}d")
    if open_w:
        notes.append("actively looking")
    if github > 60:
        notes.append(f"GitHub {github:.0f}/100")
    
    # Location note
    preferred = ['noida','pune','delhi','gurgaon','haryana','uttar pradesh','maharashtra']
    loc_lower = location.lower()
    if any(p in loc_lower for p in preferred):
        notes.append("preferred location")
    elif relocate:
        notes.append("open to relocation")
    else:
        notes.append(f"non-preferred location ({location})")
    
    # Honest concerns
    concerns = []
    if notice > 60:
        concerns.append(f"notice {notice}d")
    if response < 0.4:
        concerns.append(f"low response rate ({response:.0%})")
    if yoe > 12:
        concerns.append("overexperienced for role")
    if not expert_ir:
        concerns.append("limited direct IR/retrieval skill evidence")
    
    s2_parts = notes
    if concerns:
        s2_parts.append("concerns: " + ", ".join(concerns))
    
    s2 = "Fit signals: " + "; ".join(s2_parts) + "." if s2_parts else ""
    
    return (s1 + " " + s2).strip()[:500]


# ── Candidate loading ─────────────────────────────────────────────────────────

def load_candidates(path: str) -> dict:
    """Load candidates.jsonl → {candidate_id: candidate_dict}."""
    print(f"Loading candidates from {path}...")
    candidates = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    print(f"  Loaded {len(candidates):,} candidates.")
    return candidates


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", default=CANDIDATES_FILE,
                        help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--artifacts",  default=ARTIFACT_DIR,
                        help="Directory containing precomputed artifacts")
    parser.add_argument("--out",        default=DEFAULT_OUT,
                        help="Output CSV path")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Skip reasoning generation (faster, for testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow artifact dir override via arg
    artifact_dir = args.artifacts
    emb_path  = os.path.join(artifact_dir, "embeddings.npy")
    car_path  = os.path.join(artifact_dir, "career_features.npy")
    beh_path  = os.path.join(artifact_dir, "behavioral_features.npy")
    ids_path  = os.path.join(artifact_dir, "candidate_ids.json")
    mask_path = os.path.join(artifact_dir, "honeypot_flags.npy")

    # ── 1. Load artifacts ──────────────────────────────────────────────────────
    print("Loading precomputed artifacts...")
    embeddings     = np.load(emb_path)       # (N, 384) float32, L2-normalized
    career_feat    = np.load(car_path)        # (N, 10)  float32
    behavioral_feat= np.load(beh_path)        # (N, 10)  float32
    with open(ids_path, "r", encoding="utf-8") as _f:  # candidate_ids.json
        candidate_ids = np.array(json.load(_f), dtype=object)
    honeypot_mask  = np.load(mask_path)       # (N,) bool, True = valid

    N = embeddings.shape[0]
    print(f"  Artifacts: {N:,} candidates, {honeypot_mask.sum():,} non-valid (honeypot)")
    assert career_feat.shape    == (N, 10), f"career_features shape mismatch: {career_feat.shape}"
    assert behavioral_feat.shape== (N, 10), f"behavioral_features shape mismatch: {behavioral_feat.shape}"
    assert len(candidate_ids)   == N
    assert len(honeypot_mask)   == N

    # ── 2. Embed JD query ──────────────────────────────────────────────────────
    jd_vec = load_jd_embedding(ARTIFACT_DIR) # (384,) float32, normalized
    print(f"  JD embedding shape: {jd_vec.shape}, norm: {np.linalg.norm(jd_vec):.4f}")

    # ── 3. Score all candidates ────────────────────────────────────────────────
    print("Scoring candidates...")
    scores = compute_scores(jd_vec, embeddings, career_feat, behavioral_feat, honeypot_mask)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ── 4. Rank: top-100 with tie-breaking by candidate_id ascending ───────────
    print("Selecting top 100...")
    # Argsort descending: primary = score desc, secondary = candidate_id asc
    # We do a stable sort: first sort by candidate_id (secondary), then by score (primary).
    # np.argsort is stable, so equal scores preserve the candidate_id order.
    id_order = np.argsort(candidate_ids)           # candidate_id ascending
    scores_id_sorted   = scores[id_order]
    ids_id_sorted      = candidate_ids[id_order]

    # Now sort by score descending (stable → equal scores keep id order)
    score_rank = np.argsort(-scores_id_sorted, kind="stable")
    ranked_ids   = ids_id_sorted[score_rank]
    ranked_scores= scores_id_sorted[score_rank]

    top100_ids   = ranked_ids[:100]
    top100_scores= ranked_scores[:100]

    # Sanity: scores must be non-increasing
    assert np.all(np.diff(top100_scores) <= 1e-9), "Scores not non-increasing after sort!"
    # Sanity: all must be valid (no honeypots)
    assert np.all(top100_scores > 0), "Honeypot leaked into top 100!"

    # ── 5. Load raw candidates for reasoning ──────────────────────────────────
    if not args.no_reasoning:
        candidates_raw = load_candidates(args.candidates)
    else:
        candidates_raw = {}

    # ── 6. Write CSV ───────────────────────────────────────────────────────────
    print(f"Writing output to {args.out}...")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_pos, (cid, sc) in enumerate(zip(top100_ids, top100_scores), start=1):
            score_rounded = round(float(sc), 6)

            if args.no_reasoning:
                reasoning = ""
            else:
                c = candidates_raw.get(cid, {})
                reasoning = build_reasoning(c, sc, rank_pos)

            writer.writerow([cid, rank_pos, score_rounded, reasoning])

    print(f"\nDone. Top 100 written to: {args.out}")
    print(f"  Rank 1:   {top100_ids[0]}  score={top100_scores[0]:.6f}")
    last_idx = min(len(top100_ids), 100) - 1
    print(f"  Rank {last_idx+1}: {top100_ids[last_idx]} score={top100_scores[last_idx]:.6f}")

if __name__ == "__main__":
    main()