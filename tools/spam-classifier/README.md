# Spam Classifier demo

This folder contains a small demo for the Spam Classifier (Phase 1 baseline) and a Streamlit app that simulates a dynamic input UI similar to the referenced demo.

Files
- `train_and_save.py` - Downloads dataset, trains a TF-IDF + LogisticRegression model, saves model and vectorizer to `models/`.
- `app.py` - Streamlit interactive demo: dynamic inputs, threshold slider, top contributing features, and live prediction.
- `requirements.txt` - Python dependencies.

Quick start (Windows PowerShell)

```powershell
python -m pip install -r tools/spam-classifier/requirements.txt
# Train model (optional: the Streamlit app will train automatically if model is missing)
python tools/spam-classifier/train_and_save.py --sample-size 3000 --out-dir tools/spam-classifier/models
# Run the Streamlit app
streamlit run tools/spam-classifier/app.py
```

Notes
- The training script uses a small sample size by default to keep CI runs and local training fast. Increase `--sample-size` for a stronger model.
- The dataset is downloaded from the provided GitHub raw URL; verify licensing before publishing.

Uploading CSVs from the UI
- The Streamlit app includes an upload control in the sidebar which saves uploaded CSV files into `tools/spam-classifier/dataset/`.
- After uploading, select the CSV from the "Dataset CSV" dropdown in the sidebar and choose your label/text columns (use `col_0`/`col_1` for CSVs without headers).

