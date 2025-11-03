## Why
Spam emails remain a common nuisance and a practical classification exercise for ML. For this homework repository we want a small, reproducible ML pipeline that demonstrates data ingestion, preprocessing, model training, evaluation, and reproducible artifacts. A simple logistic regression baseline is an interpretable, fast model that is a good fit for phase 1.

## What Changes
- Add a new capability: `spam-classifier` that provides training scripts, a baseline logistic regression model, evaluation artifacts, and basic prediction CLI.
- Provide data ingestion script that downloads and prepares the dataset from the provided source.
- Add a reproducible training notebook/script and a small test harness to validate end-to-end training and prediction.
- Add CI job to run a lightweight training/evaluation smoke test.

## Data Source
Primary dataset for Phase 1:
- https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

(We should verify licensing and attribution before publication; for homework use this dataset locally.)

## Phases
### Phase 1 — Baseline model
- Train a logistic regression classifier on the provided dataset.
- Pipeline steps: download dataset → preprocessing (text cleanup) → feature extraction (TF-IDF) → train logistic regression → evaluate (precision/recall/F1, confusion matrix) → save model artifact.
- Deliverables: `tools/spam-classifier/` scripts, `notebooks/train_logistic_regression.ipynb`, and a README with run instructions.

### Phase 2 — (TBD)
- You will provide Phase 2 requirements later. Potential Phase 2 items: improved models (e.g., transformer-based), deployment as a microservice, streaming inference, or dataset augmentation.

## Impact
- New files and folders: `tools/spam-classifier/`, `notebooks/`, `ci/` (CI job), and `openspec/changes/add-spam-classifier/` (proposal/task/spec scaffolding).
- Minimal risk to existing code; this is additive.
- Testing: introduces a small training smoke test that runs quickly in CI (limit epochs / sample subset to keep runtime modest).

## Open Questions
- Should we include heavy ML dependencies (scikit-learn only) or prefer pure Node.js implementations? (Assumed: use Python + scikit-learn for Phase 1 for simplicity and reproducibility.)
- Do you want the model artifact to be saved as a pickled file, ONNX, or another format? (Assumed: pickled sklearn model is fine for local homework use.)

## Acceptance Criteria
- A script/notebook that downloads the CSV, trains a logistic regression, and outputs evaluation metrics.
- A small CLI to run `predict` against sample messages and return a spam/ham label.
- CI smoke test that runs training on a small subset and validates the pipeline completes without error.

## Next Steps
- Implement tasks listed in `tasks.md` and add a spec delta under `specs/spam-classifier/spec.md`.
- Run `openspec validate add-spam-classifier --strict` once scaffolding is in place.
