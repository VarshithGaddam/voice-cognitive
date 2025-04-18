import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_audio_files, load_audio, speech_to_text
from src.feature_extraction import extract_features
from src.modeling import detect_anomalies, calculate_risk_score
from src.visualization import save_all_plots
import pandas as pd

def save_results(features, anomaly_results, risk_scores, output_path='results/results.csv'):
    """
    Save results to a CSV file with the specified format.
    Args:
        features (dict): Extracted features.
        anomaly_results (dict): Anomaly detection results.
        risk_scores (dict): Calculated risk scores.
        output_path (str): Path to save CSV.
    Returns:
        pd.DataFrame: The saved DataFrame for visualization.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = []
    for file_name in features:
        feature_values = features[file_name]
        anomaly_value = anomaly_results[file_name]['is_anomaly']
        risk_value = risk_scores[file_name]
        row = {
            'sample_id': file_name,
            'pause_co': feature_values.get('pause_co', 0),
            'pause_avg': feature_values.get('pause_avg', 0),
            'avg_spec': feature_values.get('avg_spec', 0),
            'ra_pitch': feature_values.get('ra_pitch', 0),
            'vari': feature_values.get('vari', 0),
            'hesitation': feature_values.get('hesitation', 0),
            'lexical_div': feature_values.get('lexical_div', 0),
            'incompleteness': feature_values.get('incompleteness', 0),
            'semantic': feature_values.get('semantic', 0),
            'anomaly': 1 if anomaly_value else 0,
            'risk_score': risk_value
        }
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")
    return df

def run_pipeline(audio_dir):
    """
    Run the full pipeline for cognitive decline detection.
    Args:
        audio_dir (str): Directory containing audio files.
    Returns:
        tuple: Features, anomaly results, and risk scores.
    """
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory {audio_dir} does not exist")
        return {}, {}, {}
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Starting preprocessing...")
    processed_data = preprocess_audio_files(audio_dir)
    if not processed_data:
        print("No files processed. Exiting pipeline.")
        return {}, {}, {}
    
    for file_name, data in processed_data.items():
        if data['text']:
            try:
                with open(f'data/processed/{file_name}.txt', 'w', encoding='utf-8') as f:
                    f.write(data['text'])
                print(f"Saved transcript for {file_name}")
            except Exception as e:
                print(f"Failed to save transcript for {file_name}: {e}")
    
    print("Extracting features...")
    features = extract_features(processed_data)
    
    print("Detecting anomalies...")
    anomaly_results = detect_anomalies(features)
    
    print("Calculating risk scores...")
    risk_scores = calculate_risk_score(features, anomaly_results)
    
    print("Generating visualizations...")
    df = save_results(features, anomaly_results, risk_scores)
    save_all_plots(df, 'results/plots')
    
    print("Saving results...")
    save_results(features, anomaly_results, risk_scores)  # Ensure CSV is saved
    
    print("Pipeline completed!")
    return features, anomaly_results, risk_scores

def get_risk_score(audio_path):
    """
    API-ready function to calculate risk score for a single audio file.
    Args:
        audio_path (str): Path to audio file.
    Returns:
        float: Risk score indicating cognitive decline risk, or None if processing fails.
    """
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} does not exist")
        return None
    
    audio, sr = load_audio(audio_path)
    if audio is None:
        return None
    text = speech_to_text(audio_path)
    processed_data = {os.path.basename(audio_path): {'audio': audio, 'sr': sr, 'text': text}}
    
    features = extract_features(processed_data)
    anomaly_results = detect_anomalies(features)
    risk_scores = calculate_risk_score(features, anomaly_results)
    
    return risk_scores.get(os.path.basename(audio_path))

if __name__ == "__main__":
    run_pipeline('data/raw/')