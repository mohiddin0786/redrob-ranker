# Architecture

## Overview

`redrob-ranker` is a two-phase pipeline that ranks 100,000 candidate
profiles against the Senior AI Engineer JD and produces a top-100 CSV.

```
Phase 1: Offline precompute (Colab, T4 GPU, ~13 min)
  candidates.jsonl ──▶ embeddings.npy            (100K × 384, BGE)
                   ──▶ career_features.npy        (100K × 10)
                   ──▶ behavioral_features.npy    (100K × 10)
                   ──▶ candidate_ids.json
                   ──▶ honeypot_flags.npy

Phase 2: Online ranking (Docker, CPU-only, <5 min)
  artifacts/*.npy ──▶ rank.py ──▶ submission.csv (top 100)
```

The split exists because the compute constraints (5 min, 16GB RAM, CPU-only,
no network) make it infeasible to run the embedding model at ranking time
against 100K candidates. All model inference happens offline; the ranking
step is pure NumPy arithmetic over precomputed arrays.

## Why this split, specifically

Encoding 100K candidate profiles with `bge-small-en-v1.5` on CPU would
exceed the 5-minute budget by a wide margin. Doing it once on a Colab T4
GPU (~13 minutes) and persisting the result as a flat array means the
ranking step only needs to do a single matrix multiply against the JD query
vector — under 50ms for 100K candidates, well inside budget with room to
spare for the career/behavioral scoring.

## Scoring formula

```
final_score = 0.40 * semantic_score
            + 0.35 * career_score
            + 0.25 * behavioral_score

final_score[honeypot] = -1.0   # hard gate, applied after the weighted sum
```

These weights are frozen by design — see "Why weights are frozen" below.

### Semantic score

Raw, clipped cosine similarity between the JD query embedding and each
candidate's profile embedding. No percentile or rank-based rescaling is
applied.

This was a deliberate reversal of an earlier attempt: `rankdata`-based
percentile stretching was tried and reverted because it destroyed the
natural spread of the signal — it made marginally-different candidates look
identical or wildly different in ways unrelated to actual similarity. Raw
cosine, even though less "normalized-looking," preserved more real signal.

**Candidate text construction (`build_embed_text()`):** profile sections
are built with explicit prefixes rather than concatenated as undifferentiated
text. Skills held for 12+ months get a `"Core expertise:"` prefix; skills
with shorter duration get `"Also familiar with:"`. This stops a candidate
who added "RAG" to their skills list last week from reading as equivalent
to one with years of hands-on retrieval experience — embedding models pick
up on this framing distinction.

**JD query prefix:** the BGE asymmetric retrieval convention is followed —
the JD query is prefixed with `"Represent this sentence for searching
relevant passages:"`; candidate passages are not given a prefix. This
matches how `bge-small-en-v1.5` was trained and meaningfully affects
retrieval quality versus treating both sides symmetrically.

### Career score

```
car_scores = (career_feat @ CAREER_WEIGHTS)
             * career_feat[:, 2]   # consulting_pen
             * career_feat[:, 3]   # research_pen
             * career_feat[:, 4]   # domain_fit
```

Penalties are applied **multiplicatively**, not as additive terms summed
alongside the other career features. This is the single most important
design decision in the career scoring path.

**Why multiplicative, not additive:** the JD explicitly disqualifies certain
profiles outright — pure-research-only careers, consulting-only careers
(TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini with no product-company
experience), wrong-domain specialists (CV/speech/robotics without NLP/IR
exposure). An additive penalty (e.g. `score - 0.3` for a research-only
candidate) lets a candidate with otherwise very high career feature values
"buy back" the penalty and still surface near the top. A multiplicative
penalty (`score * 0.1`) suppresses the candidate's contribution
proportionally regardless of how strong the rest of their profile looks —
which matches how the JD actually describes these as hard disqualifiers,
not soft deductions.

**Year-of-experience scoring uses a tent function, not monotonic scaling:**
```
<2y   = 0.10
<4y   = 0.40
<5y   = 0.70
<=9y  = 1.00
<=12y = 0.60
>12y  = 0.25
```
The JD's "5-9 years" framing is explicitly a sweet spot, not a floor —
over-qualified candidates (12+ years) are penalized similarly to
under-qualified ones, reflecting the JD's stated concern about senior
engineers who've moved into pure architecture/tech-lead roles and stopped
writing production code.

### Behavioral score

```
beh_scores = behavioral_feat @ BEHAVIORAL_WEIGHTS
```

Derived from the 23 `redrob_signals` fields (activity recency, recruiter
response rate, GitHub activity, interview completion, notice period,
open-to-work flag, etc.) per `redrob_signals_doc.md`. This operationalizes
the JD's explicit instruction: *"a perfect-on-paper candidate who hasn't
logged in for 6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available."*

### Honeypot handling

```python
final[honeypot_flags] = -1.0
```

Honeypots are a **hard gate applied after scoring**, not a feature folded
into the weighted sum. A candidate flagged as a honeypot cannot recover
into the top 100 through strength elsewhere — the flag forces them to the
bottom of the ranking unconditionally, matching the ground-truth tiering
described in the submission spec (forced to relevance tier 0).

**Honeypot detection logic (final form) uses only two signals:**
1. Zero-duration "expert" proficiency skills ratio
2. Career months exceeding what years-of-experience would imply as a
   plausible maximum

An earlier version also checked for fictional company names, but this was
removed after inspection showed ~77% of the entire synthetic dataset
contains fictional company names, evenly distributed across legitimate and
honeypot profiles — making it statistically useless as a honeypot signal
and a source of false positives. The final two-signal detector identifies
30 honeypots (0.03% of the pool), which is consistent with the dataset
documentation's stated ~80 honeypots existing somewhere in the relevance
tiering (note: the candidate-pool honeypot count visible to a ranking
system need not match the full ground-truth honeypot count exactly, since
detection is necessarily a function of available features).

## Why weights are frozen (`W_SEMANTIC=0.40, W_CAREER=0.35, W_BEHAVIORAL=0.25`)

These weights were deliberately not raised or lowered in response to
specific candidate-level issues during development. The standing rule:
**if a specific domain-fit or career-matching problem shows up, fix the
feature or the embedding input that produces the symptom — never fix it by
raising `W_CAREER`.**

Raising `W_CAREER` to patch a domain-matching problem would make the
overall system increasingly keyword/feature-driven rather than driven by
the embedding model's actual semantic understanding, undermining the
system's generalizability to JDs other than this one. Every "the ranking
looks wrong here" investigation during development was resolved by
inspecting and fixing the upstream feature computation (e.g. a
`domain_fit_score()` correction) rather than by reweighting the three
top-level components.

## Tie-breaking and determinism

Final ranking uses a stable sort: score descending, `candidate_id`
ascending as the tiebreaker. This matches the submission spec's requirement
that equal scores at adjacent ranks be broken by candidate_id ascending,
and ensures the ranking is fully reproducible — same artifacts in, same
CSV out, every time, with no non-deterministic ordering.

## Known data-format gotcha (permanently fixed)

`candidates.jsonl` as shipped in the hackathon bundle is actually a single
JSON array (starts with `[`), not true line-delimited JSONL. `load_candidates()`
peeks the first non-whitespace character of the file: if it's `[`, the file
is parsed with `json.load()`; otherwise it falls back to line-by-line JSONL
parsing. This must be verified present in any new environment (including
the Docker image) before trusting any run — an environment without this
fix will either crash on load or silently parse zero candidates depending
on the parsing library used.

## What was deliberately not built

A full roadmap of possible improvements was evaluated and most were
explicitly rejected as not worth the added complexity/risk under the
hackathon's time and compute constraints:

- Multi-section semantic matching (separate embeddings per profile section,
  combined via weighted average) — added complexity without confirmed
  benefit over a single well-structured embedding text.
- Score normalization beyond raw cosine similarity (percentile, min-max,
  z-score) — tried and reverted for the semantic component specifically;
  not pursued further for the other two components either, for consistency.
- JD auto-extraction (parsing requirements out of arbitrary JDs
  automatically) — would generalize the system to unseen JDs, but out of
  scope for a single-JD hackathon submission.
- Behavioral interaction bonuses (e.g. extra bonus for high-GitHub +
  high-recruiter-response combined) — adds tuning surface area without a
  clear evaluation signal to validate against.
- Distance-based location scoring — the JD's location requirement is
  categorical (preferred city / willing-to-relocate / neither), not a
  continuous distance function, so a categorical scorer matches the actual
  requirement more directly.

Adopted from the same roadmap: structured skill-embedding text, title
hierarchy in career feature extraction, the "production ownership" language
signal (built/designed/deployed vs. assisted/supported), multiplicative
penalty structure, and explicit reasoning-string generation per candidate.
