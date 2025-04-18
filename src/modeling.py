from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def detect_anomalies(features):
    """
    Apply Isolation Forest to detect anomalous samples.
    Args:
        features (dict): Feature dictionary from feature_extraction.
    Returns:
        dict: Mapping of file names to anomaly scores and labels.
    """
    # Convert features to DataFrame
    if not features:
        return {file_name: {'anomaly_score': 0, 'is_anomaly': False} for file_name in features}
    
    feature_df = pd.DataFrame.from_dict(features, orient='index')
    feature_df = feature_df.fillna(0)  # Handle missing values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    
    # Handle small sample sizes
    if len(feature_df) < 2:
        return {file_name: {'anomaly_score': 0, 'is_anomaly': False} for file_name in features}
    
    # Dynamically adjust contamination based on sample size
    contamination = min(0.3, max(0.1, 1.0 / len(feature_df)))  # Between 0.1 and 0.3
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    scores = -model.decision_function(X)  # Negative for higher scores indicating anomalies
    
    # Map results to file names
    results = {}
    for i, file_name in enumerate(features.keys()):
        results[file_name] = {
            'anomaly_score': scores[i],
            'is_anomaly': predictions[i] == -1
        }
    
    return results

def calculate_risk_score(features, anomaly_results):
    """
    Calculate a risk score based on features and anomaly results.
    Args:
        features (dict): Feature dictionary.
        anomaly_results (dict): Anomaly detection results.
    Returns:
        dict: Mapping of file names to risk scores (0 to 1 range).
    """
    if not features or not anomaly_results:
        return {file_name: 0 for file_name in features}
    
    risk_scores = {}
    for file_name in features.keys():
        feature_values = features[file_name]
        anomaly_score = anomaly_results[file_name]['anomaly_score']
        
        # Weighted risk score with updated feature names
        weights = {
            'pause_co': 0.2,        # Higher pause count increases risk
            'pause_avg': 0.1,       # Longer pauses increase risk
            'avg_spec': -0.1,       # Higher speech rate reduces risk
            'ra_pitch': 0.1,        # Higher pitch range increases risk
            'vari': 0.1,            # Higher variance increases risk
            'hesitation': 0.2,      # More hesitations increase risk
            'lexical_div': 0.1,     # Higher lexical diversity reduces risk
            'incompleteness': 0.2,  # Higher incompleteness increases risk
            'semantic': 0.1         # Semantic issues increase risk
        }
        
        # Calculate base score from features
        base_score = sum(feature_values.get(k, 0) * w for k, w in weights.items())
        
        # Normalize anomaly score to [0, 1] and incorporate
        anomaly_weight = 0.3
        normalized_anomaly = (anomaly_score - np.min(list(anomaly_results.values())[0]['anomaly_score'])) / \
                           (np.max(list(anomaly_results.values())[0]['anomaly_score']) - np.min(list(anomaly_results.values())[0]['anomaly_score']) + 1e-10)
        total_score = base_score + (normalized_anomaly * anomaly_weight)
        
        # Normalize to [0, 1] range
        risk_scores[file_name] = max(0, min(1, (total_score + abs(np.min(list(risk_scores.values()) if risk_scores else 0))) / 2))
    
    return risk_scores