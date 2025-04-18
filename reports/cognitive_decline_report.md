Voice-Based Cognitive Decline Detection Report
Objective
This project develops a proof-of-concept pipeline to detect early indicators of cognitive decline using voice data. The pipeline processes 5–10 anonymized voice clips, extracts relevant audio and text features, and applies unsupervised machine learning to identify anomalous patterns.
Methodology

Data Preprocessing:

Audio files (WAV/MP3) were loaded using librosa.
Speech-to-text conversion was performed locally using OpenAI's Whisper model (tiny variant for speed).


Feature Extraction:

Audio Features:
Speech rate (estimated via tempo using librosa.beat.beat_track).
Pitch variability (standard deviation of fundamental frequency).
Pause duration (detected via RMS energy thresholding).


Text Features:
Pauses per sentence (approximated via hesitation markers).
Hesitation markers (e.g., "uh", "um") counted using nltk.
Sentence completion ratio (complete vs. incomplete sentences).




Modeling:

An Isolation Forest was used for unsupervised anomaly detection due to its interpretability and effectiveness with small datasets.
Features were standardized using StandardScaler.
Small sample sizes (<2 samples) are handled by assigning neutral anomaly scores.
A risk score was calculated as a weighted sum of features and anomaly scores.


Visualization:

Box plots of feature distributions (feature_trends.png).
Bar plots of anomaly scores per sample (anomaly_scores.png).


Output:

Results are saved as a CSV in results/results.csv with features, anomaly scores, and risk scores.
Transcripts are saved in data/processed/.



Key Findings

Most Insightful Features:

Pauses per sentence and hesitation count were highly indicative of cognitive stress, as they correlated strongly with anomaly scores.
Sentence completion helped identify incomplete or fragmented thoughts, a potential marker of cognitive decline.
Pitch variability showed moderate variation but was less discriminative in this small sample.


Modeling Insights:

Isolation Forest effectively identified outliers (20% contamination assumed).
Risk scores provided a simple, interpretable metric for clinical use.
Small sample handling ensured robustness for minimal datasets.



Potential Next Steps

Data Expansion: Collect a larger, clinically validated dataset with labeled cognitive decline cases.
Feature Engineering:
Incorporate prosodic features (e.g., intonation patterns).
Use advanced NLP for semantic coherence analysis.


Modeling:
Explore supervised models (e.g., XGBoost) with labeled data.
Implement time-series analysis for longitudinal voice data.


Clinical Integration:
Validate features against clinical cognitive assessments (e.g., MMSE).
Deploy the get_risk_score function as a real-time API.



Conclusion
This pipeline demonstrates the feasibility of detecting cognitive decline patterns from voice data using local processing. Pauses, hesitations, and sentence completion were the most promising features. With further refinement, this approach could support early screening in clinical settings.
Visualizations are saved as feature_trends.png and anomaly_scores.png. Results are in results/results.csv.
