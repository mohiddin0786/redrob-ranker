# Redrob Hackathon — Candidate Ranking System

Ranks a 100,000-candidate pool against the Senior AI Engineer JD and produces
a top-100 CSV submission. Built for the Intelligent Candidate Discovery &
Ranking Challenge.

## Architecture (two-phase)

1. **Offline precompute** (`precompute/`, run on Google Colab T4 GPU, ~13 min)
   Generates five artifacts from the raw candidate pool:
   - `embeddings.npy` — 100,000 × 384 BGE (`BAAI/bge-small-en-v1.5`) sentence embeddings
   - `career_features.npy — shape (100000,10), float32 engineered career signals
   - `behavioral_features.npy` — 100,000 × 10 engineered behavioral signals
   - `candidate_ids.json` — ID ordering aligned to the rows above
   - `honeypot_flags.npy` — boolean honeypot/invalid-profile flags
   - `jd_embedding.npy` — precomputed BGE embedding (384,) of the JD query,
     generated offline instead of loading the embedding model at ranking time

2. **Online ranking** (`app/rank.py`, runs in Docker, CPU-only, <5 min)
   Loads the precomputed artifacts, scores every candidate, hard-zeros
   honeypots, and writes the top-100 submission CSV. No model inference,
   no network calls, no GPU. Runtime dependency footprint is a single
   package: NumPy.

## Scoring

```
final_score = 0.40 × semantic_cosine_sim
            + 0.35 × career_score   (career_features @ career_weights,
                                      multiplied by consulting_penalty ×
                                      research_penalty × domain_fit_penalty)
            + 0.25 × behavioral_score (behavioral_features @ behavioral_weights)

honeypot candidates are assigned final_score = -1.0 and excluded from the top ranking.
Weights are hardcoded in app/rank.py
Penalty factors are multiplicative scalars in [0,1].
They reduce scores when profile attributes conflict with target requirements.
```

Semantic similarity is exact: L2-normalized BGE embeddings via a single NumPy
matrix multiply against the JD query vector — no vector database needed at
100K scale. See `docs/architecture.md` for full design rationale, including
why career/behavioral features act as correctors rather than primary
rankers, and how domain-fit and consulting penalties are applied
multiplicatively rather than as additive weight terms.

## Repo layout

```
app/rank.py                          online ranking step (this is what Docker runs)
precompute/precompute.py             offline feature/embedding generation (reference)
precompute/redrob_precompute_colab.ipynb   the actual notebook run on Colab T4
artifacts/                           precomputed .npy / .json outputs (Git LFS)
sandbox/                             small 100-candidate sample + matching artifact slice, used by the hosted Colab sandbox demo
data/                                candidates.jsonl mounted here at runtime (not committed)
docs/architecture.md                 design rationale, scoring details
docs/methodology_summary.md          short methodology summary (also in submission_metadata.yaml)
submission/team_xxx.csv              final validated submission, kept for traceability
```

## Reproducing the submission CSV

> The following commands reproduce the submitted ranking pipeline.

**Requirements:** Docker installed. No GPU, no network access needed for
this step (precompute already ran separately on Colab; its outputs are
committed to `artifacts/` via Git LFS).

Runtime execution requires no network access.
Docker image build requires dependencies to already be available.

1. Clone the repo (with Git LFS pulled — see below).
2. Place `candidates.jsonl` in a local folder, e.g. `./data/candidates.jsonl`.
3. Build the image:
   ```bash
   docker build -t redrob-ranker .
   ```
4. Run the ranking step:
   ```bash
   docker run --rm \
     -v "$(pwd)/data:/data" \
     -v "$(pwd)/submission:/app/submission" \
     redrob-ranker
   ```
   This reads `candidates.jsonl` from the mounted `/data` volume (via the
   `DATA_DIR=/data` environment variable set in the Dockerfile) and writes
   `submission.csv` to the same mounted folder.

   Note: the second `-v` mount is required — without it, the output CSV is
   written inside the ephemeral container and discarded on exit
   (`docker run --rm`).

Single reproduce command (also declared in `submission_metadata.yaml`):
```bash
docker build -t redrob-ranker . && docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/submission:/app/submission" redrob-ranker
```

Compute budget: ≤5 minutes wall-clock, ≤16GB RAM, CPU-only, no network — all
satisfied since `rank.py` only does artifact loading + NumPy scoring, no
model inference at runtime.

### Git LFS

Artifacts are tracked with Git LFS (the embeddings file alone is ~146 MB).
After cloning:
```bash
git lfs install
git lfs pull
```
If `artifacts/*.npy` show up as small text pointer files instead of binary
data after cloning, LFS didn't pull — re-run `git lfs pull` before building
the Docker image.

## Hosted sandbox demo

A small self-contained demo runs in Google Colab (link in
`submission_metadata.yaml`). It clones this repo and runs `app/rank.py`
against `sandbox/candidates_sample.jsonl` (100 real candidates) with
matching sliced artifacts, end-to-end on CPU. This is a small-sample
sanity check only, not a reflection of full ranking quality. See
`make_sandbox.py` for how the sample was generated.

## Regenerating artifacts (optional, not part of Stage 3 reproduction)

Only needed if you're modifying feature engineering or embedding logic.
Run `precompute/redrob_precompute_colab.ipynb` on a Colab T4 runtime
(~13 minutes), then download the five output artifacts into `artifacts/`,
replacing the committed versions.

## Local development

- Development was performed on Python 3.12.
  Production reproduction uses Python 3.11-slim.
  Runtime compatibility was verified.
- See `requirements.txt` for the runtime (Docker) dependency set and
  `precompute/requirements.txt` for the separate precompute dependency set
  (sentence-transformers, etc. — not needed at ranking time).