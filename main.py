from src.pipeline import run_pipeline

if __name__ == "__main__":
    features, anomaly_results, risk_scores = run_pipeline('data/raw/')
    print("Processing complete. Check results/results.csv for output.")