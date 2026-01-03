import matplotlib.pyplot as plt
import seaborn as sns

def plot_rfm(rfm, output_path):
    """
    Plot distributions of RFM metrics.
    Saves plot to output_path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(rfm['frequency'], ax=axes[0], kde=True).set_title('Frequency')
    sns.histplot(rfm['recency'], ax=axes[1], kde=True).set_title('Recency')
    sns.histplot(rfm['T'], ax=axes[2], kde=True).set_title('T')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_elbow(rfm, output_path):
    """
    Plot elbow curve for K-Means clustering.
    Saves plot to output_path.
    """
    from src.segmentation import get_elbow_data
    inertia = get_elbow_data(rfm)
    plt.plot(range(1, 8), inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.savefig(output_path)
    plt.close()

def plot_clusters(rfm, output_path):
    """
    Plot scatter of recency vs frequency, colored by cluster.
    Saves plot to output_path.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='recency', y='frequency', hue='cluster', data=rfm, palette='viridis', size='monetary_value')
    plt.title("Segments based on RFM (City-based)")
    plt.savefig(output_path)
    plt.close()

def plot_clv(rfm, output_path):
    """
    Plot distribution of CLV.
    Saves plot to output_path.
    """
    sns.histplot(rfm['CLV'], kde=True)
    plt.title("Distribution of Lifetime Value")
    plt.savefig(output_path)
    plt.close()

def plot_clv_by_cluster(rfm, output_path):
    """
    Plot average CLV per cluster.
    Saves plot to output_path.
    """
    clv_by_cluster = rfm.groupby('cluster')['CLV'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x='cluster', y='CLV', data=clv_by_cluster, palette='viridis')
    plt.title("Average CLV by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average CLV")
    plt.savefig(output_path)
    plt.close()