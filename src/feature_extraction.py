import librosa
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt', quiet=True)

def extract_audio_features(audio, sr):
    """
    Extract audio-based features (pause count, average pause, speech rate, pitch range, variance).
    Args:
        audio (np.array): Audio time series.
        sr (int): Sampling rate.
    Returns:
        dict: Audio features, or None if extraction fails.
    """
    try:
        # Pause detection (count and average duration of silent regions)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = 0.05 * np.max(rms)  # Increased threshold to reduce noise detection
        pauses = np.where(rms < silence_threshold)[0]
        pause_co = len(pauses) if len(pauses) > 0 else 0
        pause_avg = np.mean(hop_length / sr * np.diff(pauses)) if len(pauses) > 1 else 0
        
        # Speech rate (average tempo)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        avg_spec = tempo if tempo > 0 else 0
        
        # Pitch range and variance
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[magnitudes > 0]
        ra_pitch = np.ptp(pitch_values) if len(pitch_values) > 0 else 0  # Peak-to-peak range in Hz
        vari = np.var(pitch_values) if len(pitch_values) > 0 else 0  # Variance in Hz^2
        
        return {
            'pause_co': pause_co,
            'pause_avg': pause_avg,
            'avg_spec': avg_spec,
            'ra_pitch': ra_pitch,
            'vari': vari
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def extract_text_features(text):
    """
    Extract text-based features (hesitations, lexical diversity, incompleteness, semantic).
    Args:
        text (str): Transcribed text.
    Returns:
        dict: Text features.
    """
    if not text:
        return {
            'hesitation': 0,
            'lexical_div': 0,
            'incompleteness': 0,
            'semantic': 0
        }
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Hesitation markers
    hesitation_markers = ['uh', 'um', 'er', 'ah']
    hesitation = sum(1 for word in words if word in hesitation_markers)
    
    # Lexical diversity (approximated as pauses per sentence)
    lexical_div = hesitation / len(sentences) if sentences else 0
    
    # Incompleteness (inverse of sentence completion)
    complete_sentences = sum(1 for s in sentences if s.endswith(('.', '!', '?')))
    sentence_completion = complete_sentences / len(sentences) if sentences else 0
    incompleteness = 1 - sentence_completion if sentence_completion < 1 else 0
    
    # Semantic (placeholder, based on incompleteness threshold)
    semantic = 1 if incompleteness > 0.5 else 0
    
    return {
        'hesitation': hesitation,
        'lexical_div': lexical_div,
        'incompleteness': incompleteness,
        'semantic': semantic
    }

def extract_features(processed_data):
    """
    Extract features for all processed audio files.
    Args:
        processed_data (dict): Output from preprocess_audio_files.
    Returns:
        dict: Mapping of file names to feature dictionaries.
    """
    features = {}
    for file_name, data in processed_data.items():
        audio_features = extract_audio_features(data['audio'], data['sr'])
        if audio_features is None:
            print(f"Skipping feature extraction for {file_name} due to audio processing error")
            continue
        text_features = extract_text_features(data['text'])
        features[file_name] = {**audio_features, **text_features}
        print(f"Extracted features for {file_name}")
    return features