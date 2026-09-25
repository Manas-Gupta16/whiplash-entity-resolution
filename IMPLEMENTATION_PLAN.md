# Implementation Plan — Business Entity Resolution (Amazon ML Challenge 2026)

**Purpose of this doc:** a phase-by-phase execution plan an agentic coding assistant
(or a human) can work through sequentially. Each phase has a goal, concrete tasks,
the files it touches, and an acceptance check before moving on. Do not skip the
acceptance checks — a broken blocking stage or a leaky validation split invalidates
everything built on top of it.

**Constraints that apply to every phase (do not violate):**
- All TSVs read/written with `sep="\t"` explicitly.
- Never hardcode country to `{US, India}` — test set includes France, unseen in train.
  Country is an open string field everywhere in the code.
- Final model must be ≤8B params, MIT/Apache-2.0 licensed.
- No external data lookups, APIs, geocoding, or internet augmentation of any kind.
- Every Source 1 test entity must get exactly one row in `matching_results.tsv`,
  even if it has zero matches (empty string, not a missing row).
- No duplicate entity IDs within an ID list; no duplicate `source1_entity_id` rows.
- `matched_entity_ids` must be a subset of that same entity's `candidate_entity_ids`.
- Validate locally with `utils/validate_submission.py` before every leaderboard upload
  (max 5 uploads/day).

---

## Phase 0 — Repo & environment setup

**Goal:** working environment, data in place, repo skeleton created.

Tasks:
1. Create the directory structure (already scaffolded — see repo).
2. `python3 -m venv .venv && pip install -r requirements.txt`.
3. Place competition files into `data/train/` and `data/test/`.
4. Place organizer's `utils/validate_submission.py` into `utils/`.
5. `git init`, commit the skeleton, add `.gitignore` (already present — excludes
   `data/`, model artifacts, output TSVs).

**Acceptance check:** `python -c "import pandas as pd; print(pd.read_csv('data/train/train_source1.tsv', sep='\t').shape)"` runs without error.

---

## Phase 1 — EDA

**Goal:** understand the actual noise patterns before writing normalization rules —
do not guess these from the problem statement alone.

Tasks:
1. In `notebooks/eda.ipynb`: row counts per source; per-country breakdown (confirm
   France only appears in test).
2. Distribution of match-count per S1 entity in `train_ground_truth.tsv` (how many
   singletons vs 1-match vs multi-match) — this sets class balance expectations.
3. Sample ~30 true matched pairs across sources; manually diff the raw name/address
   strings. Log concrete abbreviation/typo/reordering patterns observed.
4. Check null/missing rates in `business_address` components.

**Acceptance check:** a short markdown cell in the notebook summarizing observed
noise patterns — this feeds directly into `preprocess.py`'s abbreviation maps.

---

## Phase 2 — Preprocessing & Blocking

**Scale reality check:** `train_source1` ~2.2M rows, `train_source2` ~5.0M,
`train_source3` ~5.3M; test set is similar order of magnitude. A global
nearest-neighbor / TF-IDF search over the full pool is **not viable** at this
size — too slow, too much memory. `src/blocking.py` is written around
**inverted-index lookups** (word tokens + character n-grams, both with a
max-document-frequency cutoff to drop stopword-like tokens), with similarity
reranking (rapidfuzz) applied only to the small per-entity candidate set an
index lookup returns — never as a search over the full pool.

**Goal:** normalization functions and a candidate-generation pipeline with
measured recall **and** measured runtime on a held-out validation split.

Tasks:
1. Fill in `LEGAL_SUFFIX_MAP` / `ADDRESS_ABBR_MAP` in `src/preprocess.py` from Phase 1
   findings.
2. Implement `generate_candidates()` in `src/blocking.py` following the plan already
   laid out in its docstring:
   - build `build_token_index()` and `build_ngram_index()` over the combined
     S2+S3 pool (these drop overly-common tokens/n-grams automatically per
     `config.MAX_TOKEN_DF_RATIO` / `MAX_NGRAM_DF_RATIO`)
   - for each S1 entity, union `token_candidates()` + `ngram_candidates()`
   - if the union exceeds `config.MAX_CANDIDATES_PER_ENTITY`, trim with
     `rerank_candidates()` (rapidfuzz, over the small subset only)
   - process S1 rows in batches (`config.CANDIDATE_BATCH_SIZE`), using
     `itertuples()` not `iterrows()`, and log progress — a silent multi-hour
     run with no feedback is a sign something is scaling wrong
3. Split `train_source1` entities into train/val (`config.VALIDATION_FRACTION`,
   split by `entity_id`, not by row). **Do this split on a reasonably sized
   sample first** (e.g. 20–50k S1 entities) to get a fast dev/debug loop before
   running blocking over the full multi-million-row set.
4. Run blocking on the val split; compute **blocking recall** = fraction of true
   matches (from `train_ground_truth.tsv`) that appear in the generated candidate
   set. Log this number explicitly, along with wall-clock time and peak memory
   if easily available.
5. Once the sample-scale run looks correct (recall target met, no runaway
   candidate set sizes), run it against the full training set and re-measure
   recall/runtime before moving to Phase 3.

**Acceptance check:**
- Blocking recall on validation ≥ ~0.90 (target — adjust based on what's
  achievable; if lower, add a blocking signal — e.g. a third index, or loosen
  the max-df cutoff — before moving on, since this is a hard ceiling on final
  performance).
- Candidate set size per entity stays small (roughly tens, not thousands) —
  check the distribution, not just the average, since a few pathological
  entities with huge blocks can silently dominate runtime.
- The full-scale run (2M+ S1 entities against 5M+ S2/S3 pool) completes in a
  time budget that leaves room for Phases 3–5 within the 72-hour window —
  if it doesn't, that's a signal to tighten `MAX_TOKEN_DF_RATIO`/`MAX_NGRAM_DF_RATIO`
  or optimize the indexing loop before proceeding, not something to defer.

---

## Phase 3 — Feature Engineering

**Goal:** a reusable, consistent feature vector for any (S1, candidate) pair.

Tasks:
1. Implement `pair_features()` in `src/features.py` (name/address similarity
   metrics — already stubbed with Jaccard, Levenshtein, token-sort, Jaro-Winkler).
2. Implement `build_feature_matrix()` to batch this over a candidate-pairs DataFrame.
3. Extend with any address-component features Phase 1 revealed as useful (e.g.
   city/state overlap if parseable).

**Acceptance check:** `build_feature_matrix()` runs over the full validation
candidate set without error; spot-check a few known true-match pairs and known
non-match pairs to sanity-check feature values look reasonable (matches score
high, non-matches score low).

---

## Phase 4 — Train the Matcher

**Goal:** a trained classifier and a threshold tuned specifically for macro F0.5.

Tasks:
1. In `src/train.py`: build labeled pairs from blocking output on the train split
   (positive = in ground truth, negative = blocking candidate not in ground truth
   — these are hard negatives, don't use random negatives).
2. Train LightGBM (`config.LGBM_PARAMS`) on the feature matrix.
3. Run the trained model + blocking on the **validation** split; sweep decision
   thresholds (e.g. 0.05 increments); score each with `evaluate.macro_f_beta`.
4. Pick the threshold maximizing val F0.5; hardcode it into `config.MATCH_THRESHOLD`.
5. Save the model to `config.MATCHER_PATH`.
6. Log: blocking recall, chosen threshold, val precision/recall/F0.5, feature
   importances — all needed for the methodology doc later.

**Acceptance check:** val macro F0.5 computed and printed; sanity-check that
singleton entities are mostly scoring 1.0 (correctly predicted as no-match).

---

## Phase 5 — Predict on Test & Write Submission Files

**Goal:** correctly formatted `matching_results.tsv` and `candidate_pairs.tsv`.

Tasks:
1. In `src/predict.py`: load test sources, normalize, run blocking →
   `candidate_pairs.tsv`.
2. Build features, score with the saved matcher, apply `config.MATCH_THRESHOLD`.
3. Assemble `matching_results.tsv`: **one row per S1 test entity**, comma-joined
   matched IDs, empty string for no matches, no duplicates.
4. Confirm every matched ID also appears in that entity's candidate list.

**Acceptance check:**
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```
prints `PASS`. Fix every issue before uploading.

---

## Phase 6 — Submit & Iterate

**Goal:** get on the leaderboard early, then improve within the daily submission cap.

Tasks:
1. First successful `PASS` → upload to the Portal same day (don't wait for a
   "final" version — a submitted baseline beats an unsubmitted perfect pipeline).
2. Use remaining daily submissions (max 5/day) sparingly — prefer iterating against
   your own validation F0.5 (Phase 4) over burning submissions to check ideas.
3. Track each submission's config/threshold/val-score in a simple log (e.g.
   `notebooks/submission_log.md`) — this doubles as version history for shortlisting.
4. Revisit Phase 2 blocking recall if leaderboard precision/recall looks off —
   most F0.5 ceiling problems trace back to blocking, not the matcher.

---

## Phase 7 — Robustness Pass (France / edge cases)

**Goal:** make sure nothing hardcoded breaks on unseen data.

Tasks:
1. Grep the codebase for any literal `"US"`, `"India"`, or enumerated country lists
   — remove/generalize any found.
2. Confirm blocking and features behave sanely on France-labeled test rows (no
   crashes, no silently-empty candidate sets due to a country filter).
3. Confirm every S1 test entity — regardless of country — appears in the output.

**Acceptance check:** `validate_submission.py` still passes; manually inspect a
handful of France-labeled S1 entities' predictions for plausibility.

---

## Phase 8 — Final Package Assembly

**Goal:** the `<team_name>_submission.zip` per the required structure.

Tasks:
1. Copy final `src/` into `code/business_entity_resolution/src/`.
2. Write `code/business_entity_resolution/README.md` — exact reproduce-from-scratch
   instructions (data → blocking → matching → output).
3. Write `code/business_entity_resolution/requirements.txt` (pinned versions).
4. Fill in `Documentation_template.md`: methodology, blocking strategy, model
   architecture, feature engineering, and any other relevant notes — no page limit,
   prioritize clarity and depth (this is reviewed for top teams).
5. Zip:
```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
```

**Acceptance check:** unzip fresh into a clean directory, follow only the README
instructions, and confirm both output files regenerate identically.

---

## Suggested order of work across the 72 hours

| Time | Phase |
|---|---|
| Hours 0–6 | Phase 0, 1 |
| Hours 6–16 | Phase 2 (blocking recall measured) |
| Hours 16–24 | Phase 3, 4 — get a real submission out by end of Day 1 |
| Day 2 | Phase 5, 6 iteration loop, start Phase 7 |
| Day 3 | Finish Phase 7, Phase 8 packaging, final submission |
