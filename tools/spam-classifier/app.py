import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from train_and_save import TRAIN_DATA_URL, train_and_save

import streamlit as st
from pathlib import Path
import json

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
VECT_FILE = MODEL_DIR / "vectorizer.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists() or not VECT_FILE.exists():
        return None, None, None
    model = joblib.load(MODEL_FILE)
    vect = joblib.load(VECT_FILE)
    metrics = None
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return model, vect, metrics


def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, vect, metrics = load_model()
    if model is None:
        st.info("No trained model found — training a small baseline now. This may take ~30s.")
        with st.spinner("Training baseline logistic regression..."):
            train_and_save(data_path=TRAIN_DATA_URL, out_dir=str(MODEL_DIR), sample_size=3000)
        model, vect, metrics = load_model()
    return model, vect, metrics


def explain_prediction(model, vect, text, top_k=10):
    x = vect.transform([text])
    proba = float(model.predict_proba(x)[0, 1])
    coef = model.coef_[0]
    # get feature names and contributions
    try:
        feature_names = np.array(vect.get_feature_names_out())
        contrib = x.toarray()[0] * coef
        top_idx = np.argsort(contrib)[-top_k:][::-1]
        top_feats = list(zip(feature_names[top_idx], contrib[top_idx].round(4)))
    except Exception:
        top_feats = []
    return proba, top_feats


# Helper function moved to module level for reuse
def get_series(df, selector):
    """Safely extract a series from DataFrame using column name, positional index, or preprocessing variant."""
    try:
        # Handle positional columns (col_0, col_1, etc.)
        if selector.startswith('col_'):
            idx = int(selector.split('_', 1)[1])
            if 0 <= idx < df.shape[1]:
                return df.iloc[:, idx].astype(str)
        
        # Handle actual column names if present in DataFrame
        if selector in df.columns:
            return df[selector].astype(str)
        
        # Handle preprocessing variants
        if selector.startswith('text_'):
            # Determine base text from current selection context
            base_text = None
            # Try original columns first
            if 'col_1' in df.columns:
                base_text = df['col_1'].astype(str)
            elif df.shape[1] > 1:
                base_text = df.iloc[:, 1].astype(str)
            elif 'col_0' in df.columns:
                base_text = df['col_0'].astype(str)
            elif df.shape[1] > 0:
                base_text = df.iloc[:, 0].astype(str)
            
            if base_text is not None:
                from text_preprocessing import preprocess_text
                
                # Apply the specific preprocessing step based on selector
                if selector == 'text_lower':
                    return base_text.str.lower()
                
                elif selector == 'text_contacts_masked':
                    # Replace URLs, emails, and phone numbers with tokens
                    processed = base_text.copy()
                    processed = processed.str.replace(r'https?://\S+|www\.\S+', '<URL>', regex=True)
                    processed = processed.str.replace(r'\S+@\S+\.\S+', '<EMAIL>', regex=True)
                    processed = processed.str.replace(r'\+?\d[\d\s-]{8,}', '<PHONE>', regex=True)
                    return processed
                
                elif selector == 'text_numbers':
                    # Replace digit sequences with <NUM>
                    return base_text.str.replace(r'\d+', '<NUM>', regex=True)
                
                elif selector == 'text_stripped':
                    # Remove punctuation but keep word chars, spaces, and special tokens
                    import re
                    return base_text.apply(lambda x: re.sub(r'[^\w\s<>]', ' ', x))
                
                elif selector == 'text_whitespace':
                    # Collapse repeated spaces and trim
                    return base_text.str.replace(r'\s+', ' ', regex=True).str.strip()
                
                elif selector == 'text_stopwords_removed':
                    # Remove English stopwords
                    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                    stop_words = set(ENGLISH_STOP_WORDS)
                    return base_text.apply(lambda x: ' '.join(w for w in x.split() if w.lower() not in stop_words))
                
                elif selector == 'text_clean':
                    # Apply full preprocessing pipeline
                    result = preprocess_text(base_text.iloc[0])  # Get pipeline steps
                    if 'text_clean' in result:
                        return base_text.apply(lambda x: preprocess_text(x)['text_clean'])
                
                # Fallback to original preprocessing if step not explicitly handled
                processed = base_text.apply(lambda x: preprocess_text(x)[selector])
                return processed
            
    except Exception as e:
        st.sidebar.error(f"Error processing {selector}: {str(e)}")
        return None
    
    st.sidebar.error(f"Column {selector} not found or preprocessing step not supported")
    return None

def compute_dataset_overview(df, label_selector, text_selector):
    # return class counts, token replacements (simple heuristic), and token frequencies by class
    y = None
    X_text = None

    try:
        # Get label series and convert to binary
        y_series = get_series(df, label_selector)
        if y_series is not None:
            y = y_series.map(lambda v: 1 if str(v).strip().lower() == 'spam' else 0)
        
        # Get text series
        X_text = get_series(df, text_selector)
        
        # Early return if either series is missing
        if y is None or X_text is None:
            st.sidebar.error("Failed to process selected columns")
            return None, None, None
            
    except Exception as e:
        st.sidebar.error(f"Error processing columns: {str(e)}")
        return None, None, None

    class_counts = y.value_counts().rename({0: 'ham', 1: 'spam'}) if not y.empty else None
    # token replacements simple heuristic: common tokens to mask
    masks = {'<URL>': 0, '<EMAIL>': 0, '<PHONE>': 0}
    # count occurrences in cleaned text (vectorized, naive)
    try:
        s = X_text.fillna("").astype(str)
        masks['<URL>'] = int(s.str.contains(r'http|www\.', regex=True).sum())
        masks['<EMAIL>'] = int(s.str.contains(r'@').sum())
        masks['<PHONE>'] = int(s.str.contains(r'\d').sum())
    except Exception:
        # fallback to safe iteration using items()
        masks = {'<URL>': 0, '<EMAIL>': 0, '<PHONE>': 0}
        for i, t in X_text.fillna("").items():
            if 'http' in t or 'www.' in t:
                masks['<URL>'] += 1
            if '@' in t:
                masks['<EMAIL>'] += 1
            if any(ch.isdigit() for ch in t):
                masks['<PHONE>'] += 1

    return class_counts, masks, (X_text, y)


def plot_top_tokens_by_class(vect, Xv, y, top_k=20):
    # Compute comprehensive token statistics per class.
    try:
        fn = np.array(vect.get_feature_names_out())
    except Exception:
        return None, None

    # Ensure y is a pandas Series of 0/1
    try:
        y_series = pd.Series(y).astype(int)
    except Exception:
        y_series = pd.Series(np.array(y).ravel()).astype(int)

    spam_mask = (y_series == 1).values
    ham_mask = (y_series == 0).values

    # Sum TF-IDF weights per token for each class (sparse-friendly)
    try:
        spam_tfidf = np.asarray(Xv[spam_mask].sum(axis=0)).ravel()
        ham_tfidf = np.asarray(Xv[ham_mask].sum(axis=0)).ravel()
    except Exception:
        # If boolean indexing fails, fallback to looping indices
        spam_idx_list = np.where(spam_mask)[0]
        ham_idx_list = np.where(ham_mask)[0]
        spam_tfidf = np.asarray(Xv[spam_idx_list].sum(axis=0)).ravel()
        ham_tfidf = np.asarray(Xv[ham_idx_list].sum(axis=0)).ravel()

    # Document frequency per token (how many docs contain the token) per class
    try:
        spam_dfreq = np.asarray((Xv[spam_mask] > 0).sum(axis=0)).ravel()
        ham_dfreq = np.asarray((Xv[ham_mask] > 0).sum(axis=0)).ravel()
    except Exception:
        spam_dfreq = np.asarray((Xv[spam_idx_list] > 0).sum(axis=0)).ravel()
        ham_dfreq = np.asarray((Xv[ham_idx_list] > 0).sum(axis=0)).ravel()

    # Build dataframe
    stats = pd.DataFrame({
        'token': fn,
        'spam_tfidf': spam_tfidf,
        'ham_tfidf': ham_tfidf,
        'spam_dfreq': spam_dfreq,
        'ham_dfreq': ham_dfreq,
    })

    # compute combined metrics
    stats['spam_count'] = stats['spam_dfreq']
    stats['ham_count'] = stats['ham_dfreq']

    # Log-odds using document frequency with smoothing
    alpha = 0.5
    total_spam = max(int(spam_mask.sum()), 1)
    total_ham = max(int(ham_mask.sum()), 1)
    p_spam = (stats['spam_dfreq'] + alpha) / (total_spam + alpha * 2)
    p_ham = (stats['ham_dfreq'] + alpha) / (total_ham + alpha * 2)
    # protect against p==0 or 1
    eps = 1e-9
    p_spam = p_spam.clip(eps, 1 - eps)
    p_ham = p_ham.clip(eps, 1 - eps)
    stats['log_odds'] = np.log((p_spam / (1 - p_spam)) / (p_ham / (1 - p_ham)))

    # difference metric (spam_count - ham_count)
    stats['count_diff'] = stats['spam_count'] - stats['ham_count']

    # return stats dataframe
    return stats



def main():
    # Use wide layout to better match the reference UI
    st.set_page_config(page_title="Spam Classifier Demo", layout="wide")
    st.title("Spam Classifier — Streamlit-style Demo")

    st.markdown(
        "This demo trains a small TF-IDF + Logistic Regression baseline (if needed) and lets you try dynamic inputs and thresholds."
    )

    model, vect, metrics = load_model()
    if model is None:
        if st.button("Train baseline model now"):
            model, vect, metrics = ensure_model()
            st.success("Model trained and saved.")
        else:
            st.warning("No model found. Click the button to train a baseline model (fast, small sample).")

    if model is None:
        st.stop()
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    # Sidebar: dataset + column selectors + model controls + hyperparams
    st.sidebar.markdown("#### Dataset file (optional upload)")
    uploaded = st.sidebar.file_uploader("Upload CSV to dataset/", type=["csv"] )
    if uploaded is not None:
        dest = dataset_dir / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Saved uploaded CSV to {dest.name}")

    csv_files = [p.name for p in dataset_dir.glob("*.csv")]
    #csv_files.insert(0, "(use default remote dataset)")
    csv_choice = st.sidebar.selectbox("Dataset CSV (from tools/spam-classifier/dataset)", csv_files)


    # Attempt to load selected CSV for preview and suggest columns
    df = None
    header_detected = False
    data_path = None
    if csv_choice != "(use default remote dataset)":
        data_path = dataset_dir / csv_choice
        try:
            # Try interpret with header=0 first (if file has header)
            df_hdr = pd.read_csv(data_path, header=0, encoding='utf-8')
            # heuristic: if column names look like textual headers
            hdr_names = [str(c).lower() for c in df_hdr.columns]
            # Treat as header only if common header tokens (label/message/text) appear exactly.
            header_tokens = {'label', 'message', 'text'}
            if any(name in header_tokens for name in hdr_names):
                df = df_hdr
                header_detected = True
            else:
                # fallback to header=None
                df = pd.read_csv(data_path, header=None, encoding='latin-1')
                df.columns = [f'col_{i}' for i in range(df.shape[1])]
        except Exception:
            try:
                df = pd.read_csv(data_path, header=None, encoding='latin-1')
                df.columns = [f'col_{i}' for i in range(df.shape[1])]
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")

    # Build column selector options (basic columns + preprocessing)
    preprocessing_options = [
        "text_clean",
        "text_lower",
        "text_contacts_masked",
        "text_stripped",
        "text_numbers",
        "text_whitespace",
        "text_stopwords_removed",
    ]
    
    # Only include col_0 and col_1 plus any named columns from the DataFrame
    detected_cols = ["col_0", "col_1"]  # Basic required columns
    if df is not None and header_detected:
        # Only add actual named columns if headers were detected
        named_cols = [col for col in df.columns.astype(str) 
                     if col not in detected_cols and not col.startswith('col_')]
        detected_cols.extend(named_cols)
    
    # Merge unique options preserving order: first detected columns, then preprocessing
    merged_options = []
    seen = set()
    for col in detected_cols + preprocessing_options:
        if col not in seen:
            merged_options.append(col)
            seen.add(col)

    # Suggest defaults
    if df is None:
        suggested_label = 'col_0'
        suggested_text = 'col_1'
    else:
        # prefer explicit names
        lower_cols = [c.lower() for c in df.columns.astype(str)]
        if 'label' in lower_cols:
            suggested_label = df.columns[lower_cols.index('label')]
        elif 'spam' in lower_cols:
            suggested_label = df.columns[lower_cols.index('spam')]
        else:
            suggested_label = 'col_0'

        if 'message' in lower_cols:
            suggested_text = df.columns[lower_cols.index('message')]
        elif 'text' in lower_cols:
            suggested_text = df.columns[lower_cols.index('text')]
        else:
            # if df was read with header=None we named col_1 already
            suggested_text = 'col_1'

    # Create more informative column selectors with validation
    st.sidebar.markdown("### Column Selection")
    
    # Label column selector (allow all options including preprocessing variants)
    try:
        label_index = merged_options.index(suggested_label)
    except ValueError:
        label_index = 0
    label_col = st.sidebar.selectbox(
        "Label column (should contain 'spam'/'ham')",
        merged_options,
        index=label_index,
        help="Select the column containing spam/ham labels. Can be original ('col_0') or a preprocessing variant."
    )

    # Text column selector (allow all columns)
    try:
        text_index = merged_options.index(suggested_text)
    except ValueError:
        text_index = 1
    text_col = st.sidebar.selectbox(
        "Text column (original or preprocessed)",
        merged_options,
        index=text_index,
        help="Select the message text column. Can be original ('col_1') or a preprocessing variant."
    )

    # Add validation feedback
    if df is not None:
        # Validate label column
        label_preview = get_series(df, label_col)
        if label_preview is not None:
            unique_labels = label_preview.unique()
            if not any('spam' in str(v).lower() for v in unique_labels):
                st.sidebar.warning(f"Warning: No 'spam' values found in {label_col}")
        
        # Validate text column
        text_preview = get_series(df, text_col)
        if text_preview is not None:
            avg_len = text_preview.str.len().mean()
            if avg_len < 5:  # Simple heuristic for text content
                st.sidebar.warning(f"Warning: Very short texts in {text_col} (avg {avg_len:.1f} chars)")

    # validation checks (no preview rows shown)
    if df is not None:
        if df.shape[1] < 2:
            st.sidebar.error("Selected CSV has less than 2 columns; need label and text columns.")

    # (label_col and text_col selectors are above after detection; do not duplicate them)

    model_root = Path(__file__).resolve().parent / "models"
    model_root.mkdir(exist_ok=True)
    model_dirs = [d.name for d in model_root.iterdir() if d.is_dir()]
    model_dirs.insert(0, "models")
    selected_model_dir = st.sidebar.selectbox("Model directory (under tools/spam-classifier/models)", model_dirs)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Manage model directories")
    new_model_name = st.sidebar.text_input("Create model dir (name)", value="")
    if st.sidebar.button("Create model dir") and new_model_name.strip():
        new_dir = model_root / new_model_name.strip()
        try:
            new_dir.mkdir(exist_ok=False)
            st.sidebar.success(f"Created {new_model_name}")
            model_dirs.append(new_model_name.strip())
            selected_model_dir = new_model_name.strip()
        except FileExistsError:
            st.sidebar.error("Directory already exists")

    if selected_model_dir != "models":
        if st.sidebar.button("Delete selected model dir"):
            rem = model_root / selected_model_dir
            try:
                # remove files inside then remove dir
                for p in rem.iterdir():
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        # skip nested dirs for safety
                        pass
                rem.rmdir()
                st.sidebar.success(f"Deleted {selected_model_dir}")
                selected_model_dir = "models"
            except Exception as e:
                st.sidebar.error(f"Failed to delete: {e}")

    # Hyperparameters and controls (compact ordering)
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.01)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    threshold = st.sidebar.slider("Decision threshold (spam cutoff)", 0.0, 1.0, 0.5, 0.01)
    # visual indicator for threshold: small plot with vertical red marker
    try:
        thr_fig = go.Figure()
        thr_fig.add_trace(go.Bar(x=[0.0, 1.0], y=[0, 0], marker_color=['rgba(0,0,0,0)','rgba(0,0,0,0)']))
        thr_fig.add_vline(x=threshold, line=dict(color='red', width=3))
        thr_fig.update_layout(height=60, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0,1], showticklabels=False), yaxis=dict(visible=False))
        st.sidebar.plotly_chart(thr_fig, use_container_width=True)
    except Exception:
        # ignore visualization errors in sidebar
        pass

    show_topk = st.sidebar.slider("Top contributing features", 0, 20, 8)

    # Top-token metric selector moved to sidebar for consistent control placement
    metric_options = {
        'Spam frequency (doc freq)': 'spam_count',
        'Ham frequency (doc freq)': 'ham_count',
        'Count difference (spam - ham)': 'count_diff',
        'Log-odds (spam vs ham)': 'log_odds',
        'Spam TF-IDF sum': 'spam_tfidf',
        'Ham TF-IDF sum': 'ham_tfidf',
    }
    metric_choice = st.sidebar.selectbox('Top token metric', list(metric_options.keys()), index=0)
    metric_col = metric_options[metric_choice]

    # Visualization + analysis area (attempt to compute when CSV + model are available)
    if df is not None and model is not None and vect is not None:
        class_counts, masks, xy = compute_dataset_overview(df, label_col, text_col)
        if class_counts is not None:
            st.header("Data Overview")
            # narrower right column for compact token replacements table
            c1, c2 = st.columns([1.4, 0.6])
            # class distribution
            try:
                df_counts = pd.DataFrame({ 'label': class_counts.index.astype(str), 'count': class_counts.values })
                fig = px.bar(df_counts, x='label', y='count', color='label', title='Class distribution')
                c1.plotly_chart(fig, use_container_width=True)
            except Exception:
                c1.write(class_counts)

            # token replacements / heuristics
            try:
                df_masks = pd.DataFrame([{'token': k, 'count': v} for k, v in masks.items()])
                # compact presentation: small table without large header
                c2.markdown('**Token replacements**')
                c2.table(df_masks.set_index('token'))
            except Exception:
                c2.write(masks)

        # prepare vectorized matrix and token frequency plots
        try:
            X_text, y = xy
            Xv = vect.transform(X_text.fillna(''))
            stats = plot_top_tokens_by_class(vect, Xv, y, top_k=show_topk)
            if stats is not None:
                st.subheader('Top Tokens by Class')

                # Top lists per class (use same x-axis range for alignment)
                df_ham = stats.nlargest(int(show_topk), 'ham_count')[['token', 'ham_count']]
                df_spam = stats.nlargest(int(show_topk), 'spam_count')[['token', 'spam_count']]

                # compute max for shared x-axis
                max_val = 1
                try:
                    max_val = max(df_ham['ham_count'].max() if not df_ham.empty else 0, df_spam['spam_count'].max() if not df_spam.empty else 0)
                    if max_val <= 0:
                        max_val = 1
                except Exception:
                    max_val = 1

                t1, t2 = st.columns(2)
                if not df_ham.empty:
                    fig_h = px.bar(df_ham[::-1], x='ham_count', y='token', orientation='h', title='Class: ham', labels={'ham_count':'frequency'})
                    fig_h.update_layout(xaxis=dict(range=[0, max_val * 1.05]))
                    t1.plotly_chart(fig_h, use_container_width=True)
                if not df_spam.empty:
                    fig_s = px.bar(df_spam[::-1], x='spam_count', y='token', orientation='h', title='Class: spam', labels={'spam_count':'frequency'})
                    fig_s.update_layout(xaxis=dict(range=[0, max_val * 1.05]))
                    t2.plotly_chart(fig_s, use_container_width=True)

                # Grouped comparison chart for union of top tokens
                tokens_union = pd.Index(df_ham['token']).union(df_spam['token'])
                try:
                    grp = stats.set_index('token').loc[tokens_union][['ham_count','spam_count']].reset_index()
                    grp_m = grp.melt(id_vars=['token'], value_vars=['ham_count','spam_count'], var_name='class', value_name='count')
                    grp_fig = px.bar(grp_m, x='token', y='count', color='class', barmode='group', title='Top tokens comparison')
                    grp_fig.update_layout(xaxis=dict(range=[-0.5, len(tokens_union)-0.5]), yaxis=dict(range=[0, max_val * 1.1]))
                    st.plotly_chart(grp_fig, use_container_width=True)
                except Exception:
                    pass

            # Model performance (compute on a test split using selected test_size and seed)
            X_all = Xv
            y_all = y
            # Use stratified split when possible to ensure both classes appear in train/test
            strat = None
            try:
                uniq = np.unique(y_all)
                if len(uniq) > 1:
                    strat = y_all
            except Exception:
                strat = None
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=float(test_size), random_state=int(seed), stratify=strat)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= float(threshold)).astype(int)

            st.subheader('Model Performance (Test)')
            # confusion matrix
            # Force confusion matrix to include both classes (0=ham,1=spam) so the table is stable
            try:
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            except Exception:
                cm = confusion_matrix(y_test, y_pred)

            # If cm is not 2x2 (e.g., only one class present), adapt labels dynamically
            if cm.shape == (2, 2):
                cm_df = pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1'])
            else:
                # build dynamic labels based on unique values in y_test and y_pred
                uniq = sorted(list(set(list(getattr(y_test, 'tolist', None)() if callable(getattr(y_test, 'tolist', None)) else list(y_test)) + list(y_pred))))
                cols = [f'pred_{v}' for v in uniq]
                idx = [f'true_{v}' for v in uniq]
                cm_df = pd.DataFrame(cm, index=idx, columns=cols)
            st.markdown('Confusion matrix')
            st.table(cm_df)

            # ROC and PR plots
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            pr_rec, pr_prec, _ = precision_recall_curve(y_test, y_proba)

            r1, r2 = st.columns(2)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})'))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='random'))
            roc_fig.update_layout(title='ROC', xaxis_title='FPR', yaxis_title='TPR', height=360)
            r1.plotly_chart(roc_fig, use_container_width=True)

            pr_fig = go.Figure()
            pr_fig.add_trace(go.Scatter(x=pr_rec, y=pr_prec, mode='lines', name='Precision-Recall'))
            pr_fig.update_layout(title='Precision-Recall', xaxis_title='Recall', yaxis_title='Precision', height=360)
            r2.plotly_chart(pr_fig, use_container_width=True)

            # Threshold sweep table
            thr_rows = []
            thresh_list = np.linspace(0.0, 1.0, 11)
            for t in thresh_list:
                yp = (y_proba >= t).astype(int)
                prec = precision_score(y_test, yp, zero_division=0)
                rec = recall_score(y_test, yp, zero_division=0)
                f1v = f1_score(y_test, yp, zero_division=0)
                thr_rows.append({'threshold': float(t), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1v)})
            st.markdown('Threshold sweep (precision/recall/f1)')
            thr_df = pd.DataFrame(thr_rows)
            # formatted display with fixed rounding
            try:
                st.dataframe(thr_df.style.format({'threshold':'{:.2f}', 'precision':'{:.3f}', 'recall':'{:.3f}', 'f1':'{:.3f}'}), use_container_width=True)
            except Exception:
                st.dataframe(thr_df.round({'threshold':2, 'precision':3, 'recall':3, 'f1':3}), use_container_width=True)

        except Exception as e:
            st.warning(f'Could not compute full diagnostics: {e}')

    # Live inference controls
    st.markdown('---')
    st.subheader('Live Inference')
    cspam, cham = st.columns([1, 1])
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ''
    if cspam.button('Use spam example'):
        st.session_state['input_text'] = 'Congratulations! You have won a free ticket. Click here to claim.'
    if cham.button('Use ham example'):
        st.session_state['input_text'] = "Hey, are we still on for lunch tomorrow?"

    input_text = st.text_area('Enter a message to classify', value=st.session_state.get('input_text', ''), height=120)

    show_normalized = st.checkbox("Show normalized text")

    if st.button("Predict") or input_text:
        if not input_text.strip():
            st.warning("Please enter a message to classify")
        else:
            # Get normalized text if requested
            normalized_text = ""
            if show_normalized:
                try:
                    from text_preprocessing import preprocess_text
                    normalized_text = preprocess_text(input_text)['text_clean']
                except Exception as e:
                    normalized_text = str(e)
            
            # Make prediction
            proba, top_feats = explain_prediction(model, vect, input_text, top_k=show_topk)
            label = "spam" if proba >= threshold else "ham"
            
            # Show prediction result with clear formatting
            st.success(f"Prediction: {label} | spam-prob = {proba:.4f} (threshold = {threshold:.2f})")
            
            # Show normalized text if requested
            if show_normalized:
                st.info(f"Normalized text:\n{normalized_text}")
            
            # Show probability bar
            fig = go.Figure(go.Bar(
                x=[proba],
                y=['spam probability'],
                orientation='h',
                text=[f"{proba:.2f}"],
                textposition='auto',
            ))
            fig.update_layout(
                height=100,
                margin=dict(l=0, r=20, t=0, b=0),
                xaxis=dict(range=[0, 1], showgrid=False),
                yaxis=dict(showticklabels=False),
                showlegend=False
            )
            # Add threshold line
            fig.add_vline(x=threshold, line=dict(color='black', width=1, dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature contributions if requested
            if show_topk > 0 and top_feats:
                st.markdown("#### Top contributing features (positive → spam)")
                for feat, score in top_feats:
                    st.write(f"**{feat}**: {score:+.4f}")

            if metrics:
                with st.expander("Model metrics"):
                    st.json(metrics)

    st.markdown("---")
    st.caption("This is a local demo intended to simulate the referenced Streamlit UI style and dynamic input behavior.")


if __name__ == "__main__":
    main()
