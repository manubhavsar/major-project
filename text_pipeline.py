import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix


def extract_advanced_nlp_features(text):
    """
    Extracts explicit linguistic attributes:
    1. Filler Word Ratio  (true verbal fillers only)
    2. Question Frequency  (works even without punctuation)
    3. Sentence Length Variance / Pacing
    4. Sentiment Swings (3-act polarity variance)
    """
    text = str(text)
    blob = TextBlob(text)
    words = blob.words
    total_words = max(len(words), 1)

    # ── 1. Filler Analysis (Context-Aware) ──
    # Only count TRUE verbal fillers, not legitimate words.
    # "like" as filler typically appears: start of clause, after comma,
    #   or between "I/you/we/it/he/she" and a verb.
    # We use bigram context to decide.
    true_fillers = {"um", "uh", "uhm", "umm", "hmm", "er", "literally",
                    "actually", "basically", "honestly", "obviously"}

    text_lower = text.lower()
    filler_count = 0

    # Count unambiguous fillers
    for w in words:
        if w.lower() in true_fillers:
            filler_count += 1

    # Context-aware "like" detection: only count as filler when preceded by
    # a pronoun or "was/is/be" (e.g., "I like literally", "it's like so")
    # NOT when preceded by nouns/verbs where it means "such as"
    filler_like_patterns = [
        r'\b(?:i|you|he|she|it|we|they|was|is|be|its|thats|that)\s+like\b',
        r',\s*like\b',       # "so, like, whatever"
        r'\blike\s+(?:um|uh|so|literally|really|actually)\b',
    ]
    for pat in filler_like_patterns:
        filler_count += len(re.findall(pat, text_lower))

    # Context-aware "so" detection: only filler at sentence start or after pause
    filler_so_patterns = [
        r'(?:^|[.!?])\s*so\b',   # sentence-initial "so"
        r',\s*so\s*,',            # "and, so, yeah"
    ]
    for pat in filler_so_patterns:
        filler_count += len(re.findall(pat, text_lower))

    # "you know" and "I mean" as multi-word fillers
    filler_count += text_lower.count("you know")
    filler_count += text_lower.count("i mean")
    filler_count += text_lower.count("kind of")
    filler_count += text_lower.count("sort of")

    filler_ratio = filler_count / total_words

    # ── 2. Pseudo-Sentence Splitting (handles unpunctuated ASR text) ──
    # If text has very few sentence boundaries, split on common discourse
    # markers to approximate sentence pacing.
    sentences = blob.sentences
    if len(sentences) <= 2 and total_words > 50:
        # ASR text with no punctuation — split on discourse markers
        split_markers = r'\b(?:and then|then|now|so|firstly|secondly|finally|' \
                        r'also|next|moving|another|after|thank you|' \
                        r'the next|let us|now let|moving forward)\b'
        pseudo_sents = re.split(split_markers, text_lower)
        pseudo_sents = [s.strip() for s in pseudo_sents if len(s.strip().split()) > 3]
        if len(pseudo_sents) > 2:
            sentences = [TextBlob(s) for s in pseudo_sents]

    total_sentences = max(len(sentences), 1)

    # Question detection: also catch implicit questions via question words
    num_questions = sum(1 for s in sentences if s.string.strip().endswith('?'))
    # For unpunctuated text, also detect question-word sentences
    q_words_pattern = r'\b(?:what|how|why|when|where|who|which|is it|are there|can we|do you)\b'
    for s in sentences:
        if re.search(q_words_pattern, s.string.lower()) and not s.string.strip().endswith('?'):
            num_questions += 0.5  # Partial credit for implicit questions

    q_density = num_questions / total_sentences

    # ── 3. Sentence Pacing ──
    sent_lengths = [len(s.words) if hasattr(s, 'words') else len(s.string.split())
                    for s in sentences]
    pacing_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # ── 4. Sentiment Swings ──
    chunk_size = max(1, total_words // 3)
    if total_words > 3:
        chunks = [
            " ".join(words[:chunk_size]),
            " ".join(words[chunk_size:chunk_size * 2]),
            " ".join(words[chunk_size * 2:])
        ]
        polarities = [TextBlob(c).sentiment.polarity for c in chunks]
        sentiment_swing = float(np.var(polarities))
    else:
        sentiment_swing = 0.0

    # ── 5. Readability & Lexical Diversity ──
    import textstat
    try:
        flesch = textstat.flesch_reading_ease(text)
    except:
        flesch = 50.0
    
    unique_words = set(w.lower() for w in words if w.isalpha())
    lex_div = len(unique_words) / total_words

    return [filler_ratio, q_density, pacing_var, sentiment_swing, flesch, lex_div]


def train_and_eval_models():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    import os

    os.makedirs("charts", exist_ok=True)

    print("Loading text data...")
    df = pd.read_csv("actual_text_data.csv")

    transcripts = df['transcript'].astype(str).values
    y = df['engagement'].values

    # ── Feature Group 1: TF-IDF Vocabulary Features ──
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(transcripts)

    # ── Feature Group 2: Advanced NLP Structural Features ──
    print("Extracting Advanced NLP Linguistic Features...")
    nlp_features = []
    for t in transcripts:
        nlp_features.append(extract_advanced_nlp_features(t))
    X_nlp = np.array(nlp_features)

    # Scale the NLP features before combining
    nlp_scaler = StandardScaler()
    X_nlp_scaled = nlp_scaler.fit_transform(X_nlp)

    # Combine TF-IDF (sparse) + NLP (dense)
    X_combined = hstack([X_tfidf, csr_matrix(X_nlp_scaled)])
    print(f"Combined Feature shape: {X_combined.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = []
    class_names = ["Low", "Medium", "High"]

    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump(nlp_scaler, "nlp_scaler.joblib")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        safe_name = name.replace(" ", "_").lower()
        joblib.dump(model, f"{safe_name}_text_model.joblib")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Classification Report
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        results.append({
            "Model": name, "Accuracy": acc,
            "Precision": precision, "Recall": recall, "F1-Score": f1
        })
        print(f"{name} — Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

        # Save classification report as text
        report_str = classification_report(y_test, preds, target_names=class_names)
        with open(f"charts/text_{safe_name}_report.txt", "w") as f:
            f.write(report_str)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f"Text Pipeline — {name}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"charts/text_{safe_name}_cm.png", dpi=150)
        plt.close()

    # ── Accuracy Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r['Model'] for r in results]
    accs = [r['Accuracy'] for r in results]
    bars = ax.barh(names, accs, color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Accuracy')
    ax.set_title('Text Pipeline — Model Comparison', fontweight='bold')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.2%}',
                va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("charts/text_pipeline_comparison.png", dpi=150)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv("text_pipeline_results.csv", index=False)
    print("\nText Pipeline Results saved to text_pipeline_results.csv")
    print("Confusion matrices saved to charts/")
    return results_df


if __name__ == "__main__":
    train_and_eval_models()
