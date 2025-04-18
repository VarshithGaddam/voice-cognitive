from flask import Flask, request, render_template
import os
import tempfile
import shutil
import logging
import numpy as np
import time
import librosa
from src.pipeline import load_audio
from src.feature_extraction import extract_features
from src.modeling import detect_anomalies, calculate_risk_score

app = Flask(__name__, template_folder='templates')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        transcript = request.form.get('transcript', '').strip()
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if not any(file.filename.lower().endswith(ext) for ext in ['wav', 'mp3', 'flac']):
            return render_template('index.html', error="Invalid file format. Use WAV, MP3, or FLAC")
        if not transcript:
            return render_template('index.html', error="Please provide a transcript")

        try:
            start_time = time.time()
            logger.info(f"Processing uploaded file: {file.filename}")

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                file_path = temp_file.name
            logger.info(f"Saved file to: {file_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy file to temp dir for processing
                processed_audio_path = os.path.join(temp_dir, os.path.basename(file_path))
                shutil.copy(file_path, processed_audio_path)

                logger.info(f"Loaded file for processing: {processed_audio_path}")

                # Preprocess audio
                audio, sr = load_audio(processed_audio_path)
                if audio is None:
                    logger.error("Failed to load audio file")
                    os.unlink(file_path)
                    return render_template('index.html', error="Failed to load audio file")

                # Prepare processed data
                file_basename = os.path.basename(processed_audio_path)
                processed_data = {file_basename: {'audio': audio, 'sr': sr, 'text': transcript}}
                logger.info("Prepared processed data")

                # Extract features
                features = extract_features(processed_data)
                logger.info(f"Features after extraction: {features}")
                if not features or file_basename not in features:
                    logger.error(f"Feature extraction failed for {file_basename}. Features: {features}")
                    os.unlink(file_path)
                    return render_template('index.html', error="Failed to extract features")

                # Validate features
                feature_values = features[file_basename]
                if any(np.isnan(v) or np.isinf(v) for v in feature_values.values()):
                    logger.error(f"Invalid values in features for {file_basename}: {feature_values}")
                    os.unlink(file_path)
                    return render_template('index.html', error="Invalid feature values")

                logger.info(f"Extracted features for {file_basename}")

                # Detect anomalies and calculate risk score
                anomaly_results = detect_anomalies(features)
                logger.info(f"Anomaly results: {anomaly_results}")
                if not anomaly_results or file_basename not in anomaly_results:
                    logger.warning("Anomaly detection failed for single file, using default values")
                    anomaly_results = {file_basename: {'anomaly_score': 0, 'is_anomaly': False}}

                risk_scores = calculate_risk_score(features, anomaly_results)
                logger.info(f"Risk scores: {risk_scores}")
                if file_basename not in risk_scores:
                    logger.warning("Risk score calculation failed, using default value")
                    risk_scores[file_basename] = 0

                logger.info(f"Anomaly score: {anomaly_results[file_basename]['anomaly_score']}, Risk score: {risk_scores[file_basename]}")

                # Prepare response
                result = {
                    "sample_id": os.path.splitext(file.filename)[0],
                    "pause_co": feature_values.get("pause_co", 0),
                    "pause_avg": feature_values.get("pause_avg", 0),
                    "avg_spec": feature_values.get("avg_spec", 0),
                    "ra_pitch": feature_values.get("ra_pitch", 0),
                    "vari": feature_values.get("vari", 0),
                    "hesitation": feature_values.get("hesitation", 0),
                    "lexical_div": feature_values.get("lexical_div", 0),
                    "incompleteness": feature_values.get("incompleteness", 0),
                    "semantic": feature_values.get("semantic", 0),
                    "anomaly": anomaly_results[file_basename]["is_anomaly"],
                    "risk_score": risk_scores[file_basename]
                }
                processing_time = time.time() - start_time
                logger.info(f"Response prepared: {result}")
                logger.info(f"Processing completed in {processing_time:.2f} seconds")

                # Save to CSV (optional)
                output_csv = os.path.join(temp_dir, "features_output.csv")
                import pandas as pd
                df = pd.DataFrame([feature_values])
                df["sample_id"] = file_basename
                df["anomaly"] = anomaly_results[file_basename]["is_anomaly"]
                df["risk_score"] = risk_scores[file_basename]
                df.to_csv(output_csv, index=False)
                logger.info(f"Saved results to {output_csv}")

                # Clean up temporary file
                os.unlink(file_path)

                return render_template('result.html', result=result, processing_time=processing_time)

        except Exception as e:
            logger.error(f"Internal error: {str(e)}", exc_info=True)
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
            return render_template('index.html', error=f"Internal Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)