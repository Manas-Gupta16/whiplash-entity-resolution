"""
Run the full pipeline on the TEST set and write submission files.

Pipeline:
  1. Load + normalize test_source1/2/3.tsv
  2. Run blocking (same code path as train.py) -> candidate_pairs.tsv
  3. Build features for every candidate pair
  4. Score with the trained matcher, apply config.MATCH_THRESHOLD
  5. Assemble matching_results.tsv: exactly one row per S1 test entity,
     empty matched_entity_ids for entities with no match above threshold
  6. Write both TSVs to config.OUTPUT_DIR

TODO: implement. Double check every S1 test entity gets exactly one row
even if blocking found zero candidates for it (empty match list, not a
missing row) - this is a hard rejection rule.
"""
from . import config

def main():
    raise NotImplementedError("Fill in per the plan doc, Phase 5")

if __name__ == "__main__":
    main()
