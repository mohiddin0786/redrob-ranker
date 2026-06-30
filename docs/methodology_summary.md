# Methodology Summary

Two-phase pipeline: offline precompute (Colab T4 GPU, ~13 min) generates
BGE (`bge-small-en-v1.5`) sentence embeddings, 10-dim career features, and
10-dim behavioral features for all 100K candidates. Online ranking (Docker,
CPU-only, <5 min) combines them as:

```
final = 0.40 * semantic_cosine + 0.35 * career_score + 0.25 * behavioral_score
```

with honeypots hard-zeroed (`final = -1.0`) before ranking, applied as a
gate rather than a scoring component.

**Career score** applies consulting, research, and domain-fit penalties
*multiplicatively* on top of a weighted feature dot product, rather than as
additive terms. This lets a single strong disqualifier — a CV-primary
career history, a consulting-only background with no product-company
experience — suppress a candidate regardless of how strong their other
features look, which an additive model can't reliably do. Years-of-experience
uses a tent function peaking at 5-9 years (the JD's stated sweet spot)
rather than monotonic scaling, so both under- and over-qualified candidates
are penalized.

**Semantic input** uses skill-duration thresholding: skills held 12+ months
are framed with a "Core expertise" prefix in the embedded text; shorter-duration
or recently-added skills get a weaker "Also familiar with" framing. This
reduces the degree to which keyword-stuffed profiles (skills listed without
real depth) inflate semantic similarity. No percentile/rank rescaling is
applied to the raw cosine similarity — this was tried and reverted after it
was found to destroy the natural spread of the signal.

**Search** is exact: a single NumPy matrix multiply over L2-normalized
embeddings against the JD query vector, in well under 50ms for 100K
candidates. No vector database or approximate nearest-neighbor index is
needed at this scale, and exact search avoids any ANN approximation loss.

**Honeypot detection** uses two signals only — zero-duration "expert"
proficiency skills, and career months exceeding what years-of-experience
would plausibly support. An earlier fictional-company-name check was
removed after inspection showed ~77% of the entire dataset uses fictional
company names regardless of honeypot status, making it statistically
useless as a signal.

Top-level component weights (0.40 / 0.35 / 0.25) are frozen by design:
domain-matching issues discovered during development were fixed by
correcting the underlying feature or embedding input, never by reweighting
the career component — keeping the system driven by genuine semantic
understanding rather than becoming keyword/feature-driven.
