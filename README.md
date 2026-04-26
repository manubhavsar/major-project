# Multimodal AI Engagement Predictor

An advanced, multimodal analysis tool that predicts audience engagement levels in presentations using **Linguistic Structure** (Text) and **Speech Prosody** (Audio) across 10 Machine Learning models, topped with a contextual **Gemini AI Coaching Layer**.

<img width="1024" height="1024" alt="architecture_diagram" src="https://github.com/user-attachments/assets/02a0d426-c2cc-48a1-b85a-d8c936f258b5" />

## Features

- **Verbatim ASR:** Powered by OpenAI Whisper to capture filler words (um, uh, like) for accurate verbal disfluency analysis.
- **Multimodal Pipeline:**
  - **Text Pipeline:** Analyzes Lexical Diversity, Flesch Reading Ease, Sentiment Swings, Pacing, and Question Density (trained on TED Talks).
  - **Audio Pipeline:** Analyzes MFCCs, RMS Energy, Zero-Crossing Rate, Pitch F0, Tonality (Chroma), and Timbre (Spectral Centroid) (trained on RAVDESS).
- **10-Model Consensus:** Every upload is analyzed by 10 independent models (Logistic Regression, Random Forest, SVM, Gradient Boosting, and MLP Neural Networks).
- **AI Contextual Coach:** Uses Google Gemini 1.5 Flash to read the transcript, understand context, and provide actionable coaching insights that classical ML cannot see.
- **Evaluation Dashboard:** Built-in dashboard to view model accuracy, confusion matrices, and metrics.

## Tech Stack

- **Backend:** FastAPI (Python 3.9+)
- **ASR:** OpenAI Whisper
- **ML/DS:** scikit-learn, numpy, pandas, librosa, textstat, textblob
- **LLM:** Google Gemini AI (via Google Generative AI SDK)
- **Frontend:** Modern Vanilla Vanilla HTML/JS/CSS (Professional Minimalist UI)

## Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API Key (get it free at [Google AI Studio](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/major-proj.git
   cd major-proj
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and paste your Gemini API Key.

### Running the App

1. **Start the FastAPI Server:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8080
   ```
2. **Open your browser:**
   Navigate to [http://127.0.0.1:8080](http://127.0.0.1:8080)

## Dataset Attribution

- **Text:** `rounakbanik/ted-talks` (Kaggle) - 2000+ transcripts with balanced engagement labeling.
- **Audio:** `RAVDESS` (Emotional Speech & Song) - Professional acoustic dataset for prosodic analysis.

## License

This project is licensed under the MIT License.