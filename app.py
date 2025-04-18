from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
import logging
import numpy as np
import time
import librosa

# Custom modules (adjust paths as needed)
from src.feature_extraction import count_pauses, extract_text_features, semantic_coherence
from src.modeling import run_modeling

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cognitive Decline Detection API", description="Upload an audio file and optional transcript to get cognitive decline analysis.")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), transcript: str = Form(default=None)):
    """
    Upload a single audio file (WAV, MP3, or FLAC) and optional transcript to analyze for cognitive decline.
    Returns a JSON object with feature values and risk score.
    """
    start_time = time.time()
    logger.info(f"Processing uploaded file: {file.filename or 'No filename provided'} (content_type: {file.content_type})")

    # Validate filename
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
        logger.error(f"Invalid or missing filename: {file.filename}, content_type: {file.content_type}")
        raise HTTPException(status_code=400, detail={"error": "Invalid or missing audio file"})

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, file.filename)
            logger.info(f"Attempting to save file to: {audio_path}")

            # Save uploaded audio
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"Loaded file: {audio_path}")

            # Generate transcript if not provided
            if transcript is None:
                try:
                    import whisper  # Assuming whisper is used for speech_to_text
                    model = whisper.load_model("base")  # Adjust model size as needed
                    result = model.transcribe(audio_path)
                    transcript = result["text"]
                    logger.info("Generated transcript using Whisper")
                except Exception as e:
                    logger.error(f"Failed to generate transcript: {str(e)}")
                    raise HTTPException(status_code=400, detail={"error": "Failed to generate transcript"})
            else:
                logger.info("Using provided transcript")

            # Save transcript temporarily
            text_path = os.path.join(temp_dir, file.filename.replace(".wav", ".txt"))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Feature extraction
            pause_count, pause_avg = count_pauses(audio_path)
            hesitations, lexical_div, incomplete = extract_text_features(transcript)
            semantic_sim = semantic_coherence(transcript)
            duration = librosa.get_duration(filename=audio_path)
            if duration == 0:
                logger.error("Audio file has zero duration")
                raise HTTPException(status_code=400, detail={"error": "Audio file has zero duration"})

            speech_rate = len(transcript.split()) / (duration / 60)
            pitch_variability = 0  # Placeholder; replace with actual pitch variability calculation if available

            # Prepare feature data
            file_basename = os.path.basename(file.filename)
            features = {
                file_basename: {
                    "pause_co": pause_count,
                    "pause_avg": pause_avg,
                    "avg_spec": 0,  # Placeholder; replace with actual spectral analysis if available
                    "ra_pitch": 0,  # Placeholder; replace with actual pitch range if available
                    "vari": pitch_variability,
                    "hesitation": hesitations,
                    "lexical_div": lexical_div,
                    "incompleteness": incomplete,
                    "semantic": semantic_sim
                }
            }
            logger.info(f"Features after extraction: {features}")

            # Validate features
            feature_values = features[file_basename]
            if any(np.isnan(v) or np.isinf(v) for v in feature_values.values()):
                logger.error(f"Invalid values in features for {file_basename}: {feature_values}")
                raise HTTPException(status_code=400, detail={"error": "Invalid feature values"})

            logger.info(f"Extracted features for {file_basename}")

            # Run modeling to get anomaly and risk score
            df = pd.DataFrame([{
                "sample_id": file_basename,
                "pause_count": pause_count,
                "pause_avg_duration": pause_avg,
                "speech_rate": speech_rate,
                "pitch_variability": pitch_variability,
                "hesitation_count": hesitations,
                "lexical_diversity": lexical_div,
                "incomplete_sentences": incomplete,
                "semantic_similarity": semantic_sim
            }])
            final_df = run_modeling(df)
            anomaly_score = final_df["anomaly_score"].iloc[0] if "anomaly_score" in final_df.columns else 0
            is_anomaly = bool(final_df["is_anomaly"].iloc[0]) if "is_anomaly" in final_df.columns else False
            risk_score = final_df["risk_score"].iloc[0] if "risk_score" in final_df.columns else 0

            anomaly_results = {file_basename: {"anomaly_score": anomaly_score, "is_anomaly": is_anomaly}}
            risk_scores = {file_basename: risk_score}
            logger.info(f"Anomaly results: {anomaly_results}")
            logger.info(f"Risk scores: {risk_scores}")

            # Prepare response
            response = {
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
            logger.info(f"Response types before serialization: { {k: type(v) for k, v in response.items()} }")
            logger.info(f"Response prepared: {response}")
            end_time = time.time()
            logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")

            # Save to CSV (optional, based on old code)
            output_csv = os.path.join(temp_dir, "features_output.csv")
            final_df.to_csv(output_csv, index=False)
            logger.info(f"Saved results to {output_csv}")

            return JSONResponse(content=response)

    except HTTPException as e:
        logger.error(f"HTTP Exception: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": f"Internal Server Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)