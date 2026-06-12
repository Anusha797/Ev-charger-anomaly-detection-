# AI Tool Usage Documentation

## Tools Used

 Tool -Claude

## How AI Was Used

### Architecture & Design
Claude was used to reason through the feature engineering strategy — specifically the idea of combining physics-residual features (P = V×I) with per-station Z-score normalisation. The argument for an IF + LOF ensemble over a single model was developed in dialogue with Claude, weighing global vs. local outlier detection.

### Code 
Claude generated initial boilerplate for:
- The feature pipeline (`features.py`) structure
- The argparse-based `predict.py` interface
- The training loop in `train.py`

## Where AI Was Helpful

- **Speed**: Generating boilerplate (argparse, pickle save/load, sklearn fit/transform patterns) quickly, freeing time for problem-specific logic.
- **Structure**: Proposing a clean separation of concerns (generate_data / features / eda / train / predict).
- **Articulation**: Helping translate implementation decisions into clear written rationale for the report.

 
---

## How AI-Generated Code Was Validated

1. **Read every line** before accepting — no blind copy-paste.
2. **Ran unit checks**: printed shapes, dtypes, and sample rows at each pipeline stage to confirm correctness.
3. **End-to-end smoke test**: generated data → features → train → predict → confirmed output schema and anomaly rate.
4. **Edge cases tested manually**: empty sessions, all-missing rows, unknown station IDs at inference time.
