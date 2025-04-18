import librosa
import whisper
import numpy as np
import os

def load_audio(audio_path):
    """
    Load audio file using librosa.
    Args:
        audio_path (str): Path to audio file (WAV/MP3/FLAC).
    Returns:
        tuple: Audio time series (numpy array) and sampling rate.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"Loaded {audio_path}")
        return audio, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None

def speech_to_text(audio_path):
    """
    Convert speech to text using Whisper.
    Args:
        audio_path (str): Path to audio file.
    Returns:
        str: Transcribed text or None if transcription fails.
    """
    try:
        model = whisper.load_model("tiny")  # 'tiny' for speed, use 'base' for better accuracy
        result = model.transcribe(audio_path)
        print(f"Transcribed {audio_path}")
        return result["text"].lower()
    except Exception as e:
        print(f"Whisper transcription failed for {audio_path}: {e}")
        return None

def preprocess_audio_files(audio_dir):
    """
    Preprocess all audio files in a directory.
    Args:
        audio_dir (str): Directory containing audio files.
    Returns:
        dict: Mapping of file names to (audio, sr, text).
    """
    if not os.path.exists(audio_dir):
        print(f"Error: Directory {audio_dir} does not exist")
        return {}
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    if not audio_files:
        print(f"Warning: No audio files found in {audio_dir}")
        return {}
    
    print(f"Found {len(audio_files)} audio files in {audio_dir}: {audio_files}")
    processed_data = {}
    for file_name in audio_files:
        file_path = os.path.join(audio_dir, file_name)
        audio, sr = load_audio(file_path)
        if audio is None:
            continue
        text = speech_to_text(file_path)
        processed_data[file_name] = {'audio': audio, 'sr': sr, 'text': text}
    print(f"Processed {len(processed_data)}/{len(audio_files)} files successfully")
    return processed_data