## ADDED Requirements

### Requirement: Spam Classifier Pipeline
The system SHALL provide a reproducible training pipeline that downloads the specified SMS spam dataset, preprocesses messages, extracts TF-IDF features, trains a logistic regression classifier, and emits evaluation metrics (precision, recall, F1) and a saved model artifact.

#### Scenario: Train baseline logistic regression
- **GIVEN** the dataset is available at the specified URL
- **WHEN** the training pipeline is executed on a local developer machine or CI smoke test
- **THEN** the pipeline SHALL complete without runtime errors and SHALL write a model artifact to `models/` and a metrics JSON file containing precision, recall, and F1.

#### Scenario: Predict example message
- **GIVEN** a saved model artifact and a short text message
- **WHEN** the prediction CLI is invoked with the message
- **THEN** the CLI SHALL return a label of `spam` or `ham` and a probability score between 0.0 and 1.0.

***

Notes:
- This delta introduces `spam-classifier` as a new capability intended for local development and CI smoke tests. Phase 2 will extend capabilities as requested.
