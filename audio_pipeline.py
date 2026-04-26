import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_audio_features(file_path):
    try:
        y, sr_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) \
            if np.any(magnitudes > np.median(magnitudes)) else 0
            
        # ── New Features: Chroma (Tonality) & Spectral Centroid ──
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_rate)
        chroma_mean = np.mean(chroma.T, axis=0) # 12 bands
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr_rate)
        centroid_mean = np.mean(centroid) # 1 band
        
        return np.hstack([mfccs_mean, rmse_mean, zcr_mean, pitch_mean, chroma_mean, centroid_mean])
    except Exception as e:
        print(f"Error extracting acoustic features: {e}")
        return np.zeros(29) # 13 mfcc + 1 rmse + 1 zcr + 1 pitch + 12 chroma + 1 centroid

def load_data(filepath="actual_audio_data.csv"):
    df = pd.read_csv(filepath)
    labels = df['engagement'].values

    file_paths = df['file_path'].values
    
    print("Extracting audio features... this might take a moment.")
    features = [extract_audio_features(path) for path in file_paths]
    return np.array(features), labels

def train_and_eval_models():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    import os

    os.makedirs("charts", exist_ok=True)

    print("Loading audio data...")
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature shape: {X_train_scaled.shape}")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    
    results = []
    class_names = ["Low", "Medium", "High"]
    
    joblib.dump(scaler, "audio_scaler.joblib")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(model, f"{safe_name}_audio_model.joblib")
            
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        results.append({
            "Model": name, "Accuracy": acc,
            "Precision": precision, "Recall": recall, "F1-Score": f1
        })
        print(f"{name} — Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

        report_str = classification_report(y_test, preds, target_names=class_names)
        with open(f"charts/audio_{safe_name}_report.txt", "w") as f:
            f.write(report_str)

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Greens', values_format='d')
        ax.set_title(f"Audio Pipeline — {name}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"charts/audio_{safe_name}_cm.png", dpi=150)
        plt.close()

    # ── Accuracy Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r['Model'] for r in results]
    accs = [r['Accuracy'] for r in results]
    bars = ax.barh(names, accs, color=['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Accuracy')
    ax.set_title('Audio Pipeline — Model Comparison', fontweight='bold')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.2%}',
                va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("charts/audio_pipeline_comparison.png", dpi=150)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv("audio_pipeline_results.csv", index=False)
    print("\nAudio Pipeline Results saved to audio_pipeline_results.csv")
    print("Confusion matrices saved to charts/")
    return results_df

if __name__ == "__main__":
    train_and_eval_models()
