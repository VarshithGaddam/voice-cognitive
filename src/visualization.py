import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_feature_trends(features, output_path='results/plots/feature_trends.png'):
    """
    Plot trends for extracted features.
    Args:
        features (dict): Feature dictionary.
        output_path (str): Path to save the plot.
    """
    feature_df = pd.DataFrame.from_dict(features, orient='index')
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=feature_df)
    plt.xticks(rotation=45)
    plt.title('Feature Distribution Across Samples')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_anomaly_scores(anomaly_results, output_path='results/plots/anomaly_scores.png'):
    """
    Plot anomaly scores for each sample.
    Args:
        anomaly_results (dict): Anomaly detection results.
        output_path (str): Path to save the plot.
    """
    scores = {k: v['anomaly_score'] for k, v in anomaly_results.items()}
    df = pd.DataFrame.from_dict(scores, orient='index', columns=['Anomaly Score'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y='Anomaly Score', data=df)
    plt.xticks(rotation=45)
    plt.title('Anomaly Scores for Audio Samples')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(df, output_path='results/plots/heatmap.png'):
    """
    Plot a correlation heatmap of features.
    Args:
        df (pd.DataFrame): DataFrame with features.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.drop(columns=['sample_id', 'semantic_cluster', 'anomaly']).corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_distributions(df, output_dir='results/plots'):
    """
    Plot distributions of clusters and anomalies.
    Args:
        df (pd.DataFrame): DataFrame with features.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.countplot(x="semantic_cluster", data=df)
    plt.title("Cluster Distribution")
    plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
    plt.close()

    sns.countplot(x="anomaly", data=df)
    plt.title("Anomaly Distribution")
    plt.savefig(os.path.join(output_dir, "anomaly_distribution.png"))
    plt.close()

def plot_pairwise(df, output_path='results/plots/pairplot.png'):
    """
    Plot pairwise feature comparisons.
    Args:
        df (pd.DataFrame): DataFrame with features.
        output_path (str): Path to save the plot.
    """
    selected = ["pause_avg", "avg_speech", "incompleteness", "lexical_div", "semantic_cluster"]
    sns.pairplot(df[selected], hue="semantic_cluster")
    plt.suptitle("Pairwise Feature Comparison", y=1.02)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_box(df, output_dir='results/plots'):
    """
    Plot box plots of features vs risk score.
    Args:
        df (pd.DataFrame): DataFrame with features.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    for col in ["pause_co", "incompleteness", "lexical_div"]:
        sns.boxplot(x="risk_score", y=col, data=df)
        plt.title(f"{col} vs Risk Score")
        plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
        plt.close()

def save_all_plots(df, output_dir='results/plots'):
    """
    Save all visualizations.
    Args:
        df (pd.DataFrame): DataFrame with features.
        output_dir (str): Directory to save all plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_heatmap(df, os.path.join(output_dir, "heatmap.png"))
    plot_distributions(df, output_dir)
    plot_pairwise(df, os.path.join(output_dir, "pairplot.png"))
    plot_box(df, output_dir)
    plot_feature_trends(df.set_index('sample_id').to_dict(orient='index'), os.path.join(output_dir, "feature_trends.png"))
    plot_anomaly_scores({row['sample_id']: {'anomaly_score': row['risk_score']} for row in df.to_dict(orient='records')}, os.path.join(output_dir, "anomaly_scores.png"))

# Update pipeline.py to use save_all_plots
# Note: This change is reflected in the pipeline.py update below