Voice Cognitive Decline Detection
This is a Flask-based web application designed to detect cognitive decline by analyzing uploaded audio files and their transcripts. The app processes audio features, detects anomalies, calculates risk scores, and generates output files such as CSV reports and plots for further analysis. It also supports batch processing via a main.py script and includes API testing capabilities.
Prerequisites

Python: Version 3.11.9 or higher (recommended).
Git: For version control (optional if cloning the repository).
pip: Python package manager.
Virtual Environment: Recommended for dependency management.
Audio Files: WAV, MP3, or FLAC format for processing (e.g., antonyflew_5_2.wav).

Installation

Clone the Repository (if using GitHub):
git clone https://github.com/VarshithGaddam/voice-cognitive.git
cd voice-cognitive


Create a Virtual Environment:On Windows:
python -m venv venv
.\venv\Scripts\activate

This creates and activates a virtual environment named venv.

Install Dependencies:Ensure you have the required packages by installing them from the requirements.txt file:
pip install --upgrade pip
pip install -r requirements.txt

The requirements.txt file should contain:
librosa==0.10.2.post1
openai-whisper==20240930
nltk==3.9.1
scikit-learn==1.5.1
matplotlib==3.9.2
seaborn==0.13.2
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
gunicorn==22.0.0
Flask==3.0.3

If openai-whisper==20240930 fails, install from GitHub:
pip install git+https://github.com/openai/whisper.git


Verify Installation:Check that all dependencies are installed without errors.


Running the Project

Start the Flask Application:With the virtual environment activated, run:
python app_flask.py

This starts the Flask development server on http://0.0.0.0:5000.

Access the Application:Open a web browser and go to http://localhost:5000. You should see the "Cognitive Decline Detection" page.

Usage:

Upload an audio file (e.g., antonyflew_5_2.wav in WAV, MP3, or FLAC format).
Enter the transcript of the audio manually in the provided textarea.
Click "Process" to analyze the audio and display the results.


Batch Processing with main.py:Run main.py to process all audio files in a specified directory:
python main.py

Note: Ensure a directory (e.g., audio_records/) contains your audio files and corresponding transcript files (e.g., .txt files with the same base name). main.py will automatically process each file, generate CSV and plot outputs, and save them in an output/ directory. Example structure:
audio_records/
├── antonyflew_5_2.wav
├── antonyflew_5_2.txt  # Transcript file
├── sample_1.wav
├── sample_1.txt

Customize main.py to point to your directory and handle transcripts if needed (see implementation below).


Features

Audio Analysis: Extracts features such as pause count, average pause duration, pitch variation, and lexical diversity using librosa and custom models in src/.
Anomaly Detection: Identifies potential cognitive decline indicators using scikit-learn models.
Risk Scoring: Calculates a risk score based on extracted features and anomaly results.
Output Generation:
CSV Files: Saves processed feature data, anomaly scores, and risk scores to a features_output.csv file in a temporary directory for web processing or output/ directory for batch processing.
Plots: Generates visual representations (if implemented in src/modeling.py) using matplotlib and seaborn for feature analysis or risk distribution.
Note: Web output files are stored temporarily and deleted after processing unless explicitly saved. Batch processing saves to output/.



API and Testing
API Endpoint
The app includes a /api/process endpoint for programmatic access.

Usage: Send a POST request with a multipart/form-data body containing a file (audio) and transcript (text).
Example with curl:curl -X POST -F "file=@antonyflew_5_2.wav" -F "transcript=Sample transcript text" http://localhost:5000/api/process


Response: Returns a JSON object with result (feature data) and processing_time if successful, or an error message with a 400/500 status code.

Testing Requirements

Flask Testing: Use the built-in Flask test client or a library like pytest-flask to test routes.
Install pytest and pytest-flask:pip install pytest pytest-flask


Create a tests/ directory with a test_app.py file (example below).
API Testing: Use tools like curl, Postman, or Python’s requests library to test the /api/process endpoint.
Unit Tests: Test individual functions in src/ (e.g., load_audio, extract_features) using unittest or pytest.

Example test_app.py
from app_flask import app
import pytest

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_route(client):
    rv = client.get('/')
    assert b'Cognitive Decline Detection' in rv.data

def test_api_process(client):
    with open('antonyflew_5_2.wav', 'rb') as f:
        data = {
            'file': (f, 'antonyflew_5_2.wav'),
            'transcript': (None, 'Sample transcript')
        }
        response = client.post('/api/process', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        json_data = response.get_json()
        assert 'result' in json_data

Run Tests
pytest tests/

Project Structure

app_flask.py: The main Flask application file handling file uploads, processing, and rendering.
main.py: Script for batch processing all audio records in a directory (to be implemented or customized).
templates/: Contains HTML templates (index.html and result.html) for the web interface.
src/: Directory with supporting Python modules:
pipeline.py: Loads and preprocesses audio data.
feature_extraction.py: Extracts audio and text features.
modeling.py: Detects anomalies and calculates risk scores (may include plotting logic).


requirements.txt: Lists all Python dependencies.
README.md: This file with setup, usage, and testing instructions.
output/: Directory for batch processing output (to be created if using main.py).

Output Details

CSV Output: After processing, a features_output.csv file is generated. For web use, it’s saved temporarily and deleted. For batch processing with main.py, it’s saved in the output/ directory with columns: sample_id, pause_co, pause_avg, avg_spec, ra_pitch, vari, hesitation, lexical_div, incompleteness, semantic, anomaly, and risk_score.
Plot Output: If implemented in src/modeling.py, graphs (e.g., feature distributions or anomaly scores) are generated using matplotlib and seaborn. These are displayed in the browser for web use or saved as .png files in output/ for batch processing. Check src/modeling.py for customization.
Location: Web outputs are in a system-generated temporary directory (e.g., C:\Users\Varshith\AppData\Local\Temp). Batch outputs are in output/.



Note: Modify process_audio to match your app_flask.py logic or integrate it properly. This is a basic example.
Troubleshooting

Dependency Errors: If a package fails to install, ensure your Python version is compatible (e.g., downgrade to 3.11.9 if using 3.12+). Check the error message and update requirements.txt.
App Not Running: Verify the port (5000) is free and that app_flask.py has no syntax errors. Run with python -m flask run as an alternative.
Audio Processing Issues: Ensure uploaded files are valid and src/ modules are implemented. Test with a small audio file.
Output Missing: Check app_flask.py logs for errors and ensure matplotlib/seaborn are configured for saving plots.
Batch Processing Errors: Verify audio_records/ directory and transcript files exist. Adjust main.py for your data structure.

Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Report issues via GitHub Issues.
License
MIT License - See LICENSE for details.
