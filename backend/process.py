import os
import joblib
import re
import numpy as np
import librosa
from moviepy import VideoFileClip
import speech_recognition as sr
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix
from dotenv import load_dotenv

# Load environment variables from .env file in the PROJECT ROOT
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

MODELS = {}

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Support Vector Machine",
    "Gradient Boosting",
    "MLP Classifier"
]


def get_safe_name(name):
    return name.replace(" ", "_").lower()


import whisper

def load_models():
    if not MODELS:
        print("Loading all 10 ML models for inference & Whisper ASR...")
        try:
            # Load Whisper ASR
            MODELS['whisper'] = whisper.load_model("base")
            print("Whisper Base Model loaded.")
        except Exception as e:
            print("Could not load Whisper model:", e)
        
        try:
            MODELS['tfidf_vectorizer'] = joblib.load('../tfidf_vectorizer.joblib')
            MODELS['nlp_scaler'] = joblib.load('../nlp_scaler.joblib')
            for name in MODEL_NAMES:
                safe = get_safe_name(name)
                MODELS[f'text_{safe}'] = joblib.load(f'../{safe}_text_model.joblib')
        except Exception as e:
            print("Could not load text models:", e)

        try:
            MODELS['audio_scaler'] = joblib.load('../audio_scaler.joblib')
            for name in MODEL_NAMES:
                safe = get_safe_name(name)
                MODELS[f'audio_{safe}'] = joblib.load(f'../{safe}_audio_model.joblib')
        except Exception as e:
            print("Could not load audio models:", e)
        print("Models loaded successfully.")


# ─── Text Feature Extraction (mirrors text_pipeline.py exactly) ───

def extract_advanced_nlp_features(text):
    """Context-aware linguistic feature extraction."""
    text = str(text)
    blob = TextBlob(text)
    words = blob.words
    total_words = max(len(words), 1)

    # 1. Context-Aware Filler Detection
    true_fillers = {"um", "uh", "uhm", "umm", "hmm", "er", "literally",
                    "actually", "basically", "honestly", "obviously"}
    text_lower = text.lower()
    filler_count = sum(1 for w in words if w.lower() in true_fillers)

    filler_like_patterns = [
        r'\b(?:i|you|he|she|it|we|they|was|is|be|its|thats|that)\s+like\b',
        r',\s*like\b',
        r'\blike\s+(?:um|uh|so|literally|really|actually)\b',
    ]
    for pat in filler_like_patterns:
        filler_count += len(re.findall(pat, text_lower))

    filler_so_patterns = [
        r'(?:^|[.!?])\s*so\b',
        r',\s*so\s*,',
    ]
    for pat in filler_so_patterns:
        filler_count += len(re.findall(pat, text_lower))

    filler_count += text_lower.count("you know")
    filler_count += text_lower.count("i mean")
    filler_count += text_lower.count("kind of")
    filler_count += text_lower.count("sort of")

    filler_ratio = filler_count / total_words

    # 2. Pseudo-Sentence Splitting for ASR text
    sentences = blob.sentences
    if len(sentences) <= 2 and total_words > 50:
        split_markers = r'\b(?:and then|then|now|so|firstly|secondly|finally|' \
                        r'also|next|moving|another|after|thank you|' \
                        r'the next|let us|now let|moving forward)\b'
        pseudo_sents = re.split(split_markers, text_lower)
        pseudo_sents = [s.strip() for s in pseudo_sents if len(s.strip().split()) > 3]
        if len(pseudo_sents) > 2:
            sentences = [TextBlob(s) for s in pseudo_sents]

    total_sentences = max(len(sentences), 1)

    num_questions = sum(1 for s in sentences if s.string.strip().endswith('?'))
    q_words_pattern = r'\b(?:what|how|why|when|where|who|which|is it|are there|can we|do you)\b'
    for s in sentences:
        if re.search(q_words_pattern, s.string.lower()) and not s.string.strip().endswith('?'):
            num_questions += 0.5

    q_density = num_questions / total_sentences

    # 3. Sentence Pacing
    sent_lengths = [len(s.words) if hasattr(s, 'words') else len(s.string.split())
                    for s in sentences]
    pacing_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # 4. Sentiment Swings
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

    # 5. Readability & Lexical Diversity
    import textstat
    try:
        flesch = textstat.flesch_reading_ease(text)
    except:
        flesch = 50.0
    
    unique_words = set(w.lower() for w in words if w.isalpha())
    lex_div = len(unique_words) / total_words

    return [filler_ratio, q_density, pacing_var, sentiment_swing, flesch, lex_div]


# ─── Audio Feature Extraction (mirrors audio_pipeline.py exactly) ───

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
        print("Error extracting acoustic features:", e)
        return np.zeros(29)


# ─── Media Helpers ───

def extract_audio_from_video(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_audio_path, codec='pcm_s16le', fps=16000, logger=None
        )
        return True
    except Exception as e:
        print("Error extracting audio:", e)
        return False


def transcribe_audio(audio_path):
    """
    Transcription using local OpenAI Whisper.
    We pass a specific initial prompt to encourage Whisper to NOT
    sanitize the transcript, preserving 'ums', 'uhs', and stutters.
    """
    try:
        if 'whisper' not in MODELS:
             MODELS['whisper'] = whisper.load_model("base")
             
        print("Running Whisper ASR via Native Memory...")
        model = MODELS['whisper']
        
        # Bypass Whisper's reliance on the system ffmpeg binary by feeding raw numpy arrays
        # Whisper strictly requires audio at exactly 16000 Hz sample rate
        y, _ = librosa.load(audio_path, sr=16000)
        
        # We prime it with a prompt full of disfluencies so it outputs them
        result = model.transcribe(
            y, 
            initial_prompt="Umm, let me think, uh, like, so obviously..."
        )
        return result['text'].strip()
    except Exception as e:
        print("Transcription failed:", e)
        return ""


# ─── Main Prediction Loop ───

def run_predictions(file_path):
    load_models()

    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi'))
    audio_path = file_path

    if is_video:
        audio_path = file_path.rsplit('.', 1)[0] + ".wav"
        if not extract_audio_from_video(file_path, audio_path):
            raise Exception("Failed to process video audio")

    transcript = transcribe_audio(audio_path)
    engagement_map = {0: "Low", 1: "Medium", 2: "High", -1: "Unknown"}

    text_results = []
    audio_results = []
    text_preds_num = []
    audio_preds_num = []

    # ── Compute NLP feature values for display ──
    nlp_feature_values = {}
    if transcript:
        raw_nlp = extract_advanced_nlp_features(transcript)
        nlp_feature_values = {
            "Filler Word Ratio": f"{raw_nlp[0]:.4f}",
            "Question Density": f"{raw_nlp[1]:.4f}",
            "Pacing Variance": f"{raw_nlp[2]:.2f}",
            "Sentiment Swing": f"{raw_nlp[3]:.6f}",
            "Flesch Reading Ease": f"{raw_nlp[4]:.2f}",
            "Lexical Diversity": f"{raw_nlp[5]:.4f}",
        }

    # ── Text Inference ──
    if transcript and 'tfidf_vectorizer' in MODELS and 'nlp_scaler' in MODELS:
        tfidf_vec = MODELS['tfidf_vectorizer'].transform([transcript])
        nlp_raw = np.array([extract_advanced_nlp_features(transcript)])
        nlp_scaled = MODELS['nlp_scaler'].transform(nlp_raw)
        X_combined = hstack([tfidf_vec, csr_matrix(nlp_scaled)])

        for name in MODEL_NAMES:
            key = f'text_{get_safe_name(name)}'
            if key in MODELS:
                m = MODELS[key]
                pred = int(m.predict(X_combined)[0])
                conf = 0.0
                if hasattr(m, 'predict_proba'):
                    proba = m.predict_proba(X_combined)[0]
                    conf = float(max(proba))
                text_preds_num.append(pred)
                text_results.append({
                    "model": name,
                    "prediction": engagement_map.get(pred, "Unknown"),
                    "confidence": f"{conf:.1%}"
                })

    # ── Compute Audio feature values for display ──
    audio_raw_features = extract_audio_features(audio_path)
    audio_feature_values = {
        "MFCC Band Mean": f"{np.mean(audio_raw_features[:13]):.4f}",
        "RMS Energy": f"{audio_raw_features[13]:.4f}",
        "Zero-Crossing Rate": f"{audio_raw_features[14]:.6f}",
        "Pitch F0 (Hz)": f"{audio_raw_features[15]:.2f}",
        "Tonality/Chroma": f"{np.mean(audio_raw_features[16:28]):.4f}",
        "Spectral Centroid": f"{audio_raw_features[28]:.2f}",
    }

    # ── Audio Inference ──
    if transcript and 'audio_scaler' in MODELS:
        features_scaled = MODELS['audio_scaler'].transform([audio_raw_features])
        for name in MODEL_NAMES:
            key = f'audio_{get_safe_name(name)}'
            if key in MODELS:
                m = MODELS[key]
                pred = int(m.predict(features_scaled)[0])
                conf = 0.0
                if hasattr(m, 'predict_proba'):
                    proba = m.predict_proba(features_scaled)[0]
                    conf = float(max(proba))
                audio_preds_num.append(pred)
                audio_results.append({
                    "model": name,
                    "prediction": engagement_map.get(pred, "Unknown"),
                    "confidence": f"{conf:.1%}"
                })

    # ── Consensus ──
    avg_text = int(np.round(np.mean(text_preds_num))) if text_preds_num else -1
    avg_audio = int(np.round(np.mean(audio_preds_num))) if audio_preds_num else -1
    if avg_text != -1 and avg_audio != -1:
        consensus = int(np.round((avg_text + avg_audio) / 2))
    else:
        consensus = max(avg_text, avg_audio)

    # ── Find highest confidence model across both pipelines ──
    all_model_results = text_results + audio_results
    best_model = max(all_model_results, key=lambda x: float(x['confidence'].strip('%'))/100) if all_model_results else None
    best_model_info = f"{best_model['model']} predicted {best_model['prediction']} with {best_model['confidence']} confidence" if best_model else "No models ran"

    # ── Gemini AI Contextual Report ──
    gemini_report = generate_gemini_report(
        transcript,
        nlp_feature_values,
        audio_feature_values,
        engagement_map.get(avg_text, "Unknown"),
        engagement_map.get(avg_audio, "Unknown"),
        engagement_map.get(consensus, "Unknown"),
        best_model_info
    )

    return {
        "text_pipeline": {
            "attributes": [
                "TF-IDF Vocabulary (3000 terms)",
                "Filler Word Ratio (context-aware)",
                "Question Frequency",
                "Sentence Length Variance (Pacing)",
                "Sentiment Swings (3-Act Arc)",
                "Flesch Reading Ease",
                "Lexical Diversity"
            ],
            "feature_values": nlp_feature_values,
            "models": text_results,
            "consensus": engagement_map.get(avg_text, "Unknown")
        },
        "audio_pipeline": {
            "attributes": [
                "13-Band MFCCs",
                "RMS Energy (Loudness)",
                "Zero-Crossing Rate (Voice Texture)",
                "Pitch F0",
                "Tonality (Chroma)",
                "Spectral Centroid (Timbre)"
            ],
            "feature_values": audio_feature_values,
            "models": audio_results,
            "consensus": engagement_map.get(avg_audio, "Unknown")
        },
        "overall_engagement": engagement_map.get(consensus, "Unknown"),
        "transcript": transcript if transcript else "No spoken words detected.",
        "gemini_report": gemini_report
    }


# ─── Gemini AI Contextual Report Generation ───

def generate_gemini_report(transcript, nlp_values, audio_values, text_consensus, audio_consensus, overall, best_model_info=""):
    """
    Uses Google Gemini (2.5 Flash) to deeply analyze the transcript and computed features.
    """
    import google.generativeai as genai
    import json

    # Try environment variable first (best for GitHub/Deployment)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return {"summary": "Gemini Key missing. Please set GEMINI_API_KEY in your .env file.", "engagement_rating": "N/A"}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""Expert presentation coach analysis.
TRANSCRIPT: {transcript}
METRICS: Linguistic({nlp_values}), Acoustic({audio_values})
BEST ML MODEL: {best_model_info}

Analyze content for engagement. Respond in RAW JSON:
{{
    "summary": "2-3 sentences",
    "engagement_rating": "Low/Medium/High",
    "strengths": ["list"],
    "improvements": ["list"],
    "coaching_tips": ["list"],
    "filler_analysis": "detail",
    "structure_analysis": "detail",
    "confidence_score": "0-100%"
}}"""

        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        return json.loads(raw_text)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            summary = "Gemini Rate Limit (429) hit. Free tier allows 15 requests/min. Please wait 60 seconds and try again."
        else:
            summary = f"Gemini API Error: {error_msg}"
            
        print(f"DEBUG: Gemini Failure: {error_msg}")
        return {
            "summary": summary,
            "engagement_rating": "N/A",
            "strengths": ["API is temporarily throttled or key is invalid."],
            "improvements": ["Try a shorter clip or wait a minute."],
            "coaching_tips": ["Check your Gemini API Dashboard for quota usage."],
            "filler_analysis": "N/A",
            "structure_analysis": "N/A",
            "confidence_score": "0%"
        }
