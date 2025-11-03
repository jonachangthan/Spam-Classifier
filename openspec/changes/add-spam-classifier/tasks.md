## 1. Implementation
- [ ] 1.1 Create `tools/spam-classifier/` with:
  - `download_data.py` (or `.ts`): downloads CSV from the provided URL and saves to `data/`.
  - `preprocess.py`: basic text cleaning and label extraction.
  - `train.py`: trains a logistic regression model using scikit-learn and saves the artifact to `models/`.
  - `predict.py`: simple CLI that loads saved model and predicts label for input text.
- [ ] 1.2 Add `notebooks/train_logistic_regression.ipynb` for exploratory work and reproducible runs.
- [ ] 1.3 Add example `data/sample.csv` (small subset) for CI and quick tests.
- [ ] 1.4 Add unit tests for preprocessing and prediction logic.
- [ ] 1.5 Document usage in `tools/spam-classifier/README.md` and update `openspec/project.md` if needed.

## 2. Validation
- [ ] 2.1 Add spec delta under `openspec/changes/add-spam-classifier/specs/spam-classifier/spec.md`.
- [ ] 2.2 Run `openspec validate add-spam-classifier --strict` and fix any formatting issues.

## 3. CI
- [ ] 3.1 Add CI job `ci/train-smoke-test` that runs `download_data.py` on a small sample and runs `train.py` with limited data to ensure pipeline completes.

## 4. Documentation
- [ ] 4.1 Add README instructions for running training locally and explanation of dataset source and license.

## 5. Handoff
- [ ] 5.1 Prepare PR that links to this OpenSpec change and requests review before implementation completes.
