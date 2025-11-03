# Spam Classifier Demo

An interactive Streamlit application for SMS/Email spam classification using TF-IDF and Logistic Regression.

## Features

- 🔍 Interactive text preprocessing visualization
- 📊 Real-time model predictions with probability scores
- 📈 Model performance metrics and visualizations
- 🎯 Adjustable classification threshold
- 🔄 Dynamic text preprocessing options
- 📋 Support for custom CSV datasets

## Project Structure

```
hw3/
├── tools/
│   └── spam-classifier/
│       ├── app.py                  # Streamlit web interface
│       ├── train_and_save.py       # Model training script
│       ├── text_preprocessing.py    # Text preprocessing pipeline
│       ├── requirements.txt        # Python dependencies
│       ├── README.md              # Tool documentation
│       ├── dataset/               # Data storage
│       │   ├── PREPROCESSING.md   # Preprocessing documentation
│       │   └── sms_spam_clean.csv # Processed dataset
│       └── models/                # Model artifacts
│           ├── metrics.json       # Training metrics
│           ├── model.joblib       # Trained classifier
│           └── vectorizer.joblib  # TF-IDF vectorizer
└── README.md                      # Main documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hw3
```

2. Install dependencies:
```bash
pip install -r tools/spam-classifier/requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run tools/spam-classifier/app.py
```

2. Train a new model (optional):
```bash
python tools/spam-classifier/train_and_save.py --sample-size 3000
```

3. Train with preprocessing steps saved:
```bash
python tools/spam-classifier/train_and_save.py --sample-size 3000 --steps-out-dir tools/spam-classifier/dataset/preprocessing_steps --save-step-columns
```

## Text Preprocessing Pipeline

The application includes a comprehensive text preprocessing pipeline:

1. **Lowercase Normalization** (`text_lower`)
   - Converts all text to lowercase
   - Example: "Ok lar... Joking" → "ok lar... joking"

2. **Contact Masking** (`text_contacts_masked`)
   - Replaces URLs, emails, and phone numbers with tokens
   - Tokens: `<URL>`, `<EMAIL>`, `<PHONE>`

3. **Number Handling** (`text_numbers`)
   - Replaces digit sequences with `<NUM>`
   - Example: "Text FA to 87121" → "text fa to <NUM>"

4. **Punctuation Stripping** (`text_stripped`)
   - Removes punctuation while preserving word characters and special tokens
   - Example: "crazy.. available" → "crazy available"

5. **Whitespace Normalization** (`text_whitespace`)
   - Collapses repeated spaces and trims
   - Standardizes spacing between words

6. **Optional Stopword Removal** (`text_stopwords_removed`)
   - Removes common English stopwords
   - Example: "this is a test" → "test"

## Model Details

- **Vectorizer**: TF-IDF with sublinear scaling
  - Max features: 5000
  - N-gram range: (1, 2)
  - Min document frequency: 2

- **Classifier**: Logistic Regression
  - Balanced class weights
  - C=2.0 regularization
  - Max iterations: 1000

## Interactive Features

1. **Dataset Selection**
   - Upload custom CSV files
   - Select columns for label/text
   - Choose preprocessing variants

2. **Model Analysis**
   - Class distribution visualization
   - Token replacement statistics
   - Top tokens by class analysis

3. **Performance Metrics**
   - ROC curve with AUC
   - Precision-Recall curve
   - Confusion matrix
   - Threshold sweep analysis

4. **Live Inference**
   - Real-time text classification
   - Probability visualization
   - Feature contribution analysis
   - Normalized text preview

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)