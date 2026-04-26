import os
import numpy as np
import pandas as pd
import soundfile as sf

def generate_mock_text_data(num_samples=150):
    """Generates mock transcripts and engagement labels."""
    np.random.seed(42)
    
    low_eng_phrases = ["this is boring", "i don't understand", "not very good", "very slow today", "i am confused"]
    med_eng_phrases = ["this is okay", "makes sense", "moving along", "decent points", "standard presentation"]
    high_eng_phrases = ["wow this is amazing", "great point", "very interesting", "i love this", "excellent pacing"]
    
    data = []
    labels = []
    
    for _ in range(num_samples // 3):
        # Low engagement
        data.append(np.random.choice(low_eng_phrases) + " " + np.random.choice(low_eng_phrases))
        labels.append(0) # Low
        
        # Medium engagement
        data.append(np.random.choice(med_eng_phrases) + " " + np.random.choice(med_eng_phrases))
        labels.append(1) # Medium
        
        # High engagement
        data.append(np.random.choice(high_eng_phrases) + " " + np.random.choice(high_eng_phrases))
        labels.append(2) # High
        
    df = pd.DataFrame({'transcript': data, 'engagement': labels})
    return df

def generate_mock_audio_data(output_dir="mock_audio", num_samples=150):
    """Generates mock audio files (wav) and labels."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    sample_rate = 22050
    
    file_paths = []
    labels = []
    
    for i in range(num_samples):
        label = i % 3
        duration = 1.0 + np.random.rand() # 1 to 2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # generate different 'frequencies' / 'energy' based on label just as a mock pattern
        if label == 0:
            freq = 100 + np.random.rand() * 50
            amp = 0.2
        elif label == 1:
            freq = 300 + np.random.rand() * 100
            amp = 0.5
        else:
            freq = 600 + np.random.rand() * 200
            amp = 0.9
            
        audio_signal = amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.01, len(t))
        
        file_path = os.path.join(output_dir, f"sample_{i}_label_{label}.wav")
        sf.write(file_path, audio_signal, sample_rate)
        
        file_paths.append(file_path)
        labels.append(label)
        
    df = pd.DataFrame({'file_path': file_paths, 'engagement': labels})
    return df

if __name__ == "__main__":
    print("Generating mock text data...")
    text_df = generate_mock_text_data()
    text_df.to_csv("mock_text_data.csv", index=False)
    print("Saved mock_text_data.csv")
    
    print("Generating mock audio data...")
    audio_df = generate_mock_audio_data()
    audio_df.to_csv("mock_audio_data.csv", index=False)
    print("Saved mock_audio_data.csv in mock_audio/")
