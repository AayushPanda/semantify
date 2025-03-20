import umap
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive use

def reduce(embeddings, target_dims, method="UMAP"):
    """Reduces dimensionality of embeddings using given method."""
    n_samples = len(embeddings)
    
    # Ensure we have enough samples for the requested dimensions
    if n_samples <= target_dims + 1:  # Need at least target_dims + 2 samples
        # Set target_dims to be safely smaller than n_samples
        target_dims = max(2, n_samples - 2)
        
    if method == "TSNE":
        # For t-SNE, perplexity must be less than n_samples
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=target_dims, perplexity=perplexity, n_iter=300)
        return tsne.fit_transform(embeddings)
    else:  # UMAP
        # For UMAP, set both n_components and n_neighbors appropriately
        n_neighbors = min(15, n_samples - 1)
        
        try:
            # First attempt with requested dimensions
            return umap.UMAP(
                n_components=target_dims,
                n_neighbors=n_neighbors,
                min_dist=0.2
            ).fit_transform(embeddings)
        except (ValueError, TypeError) as e:
            safe_dims = max(2, n_samples // 2 - 1)  # really stupid but a workaround k>=n issue
            print(f"Warning: Reducing dimensions from {target_dims} to {safe_dims} due to insufficient data")
            return umap.UMAP(
                n_components=safe_dims,
                n_neighbors=n_neighbors,
                min_dist=0.2
            ).fit_transform(embeddings)

# TODO: try HDBSCAN for clustering for speed
# from cuml.cluster import HDBSCAN -- RAPIDS for GPU-accelerated clustering
def cluster(embeddings, thres=2.5):
    """Applies UMAP for dimensionality reduction and Agglomerative Clustering."""
    if len(embeddings[0]) > 30 and len(embeddings) > 30:
        clusterable_embedding = reduce(embeddings, 30)
    else:
        clusterable_embedding = embeddings

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=thres)
    labels = clustering.fit_predict(clusterable_embedding)

    return labels, clusterable_embedding

def ncluster(embeddings, n=10):
    """Applies UMAP for dimensionality reduction and Agglomerative Clustering."""
    if len(embeddings[0]) > 30 and len(embeddings) > 30:
        clusterable_embedding = reduce(embeddings, 30)
    else:
        clusterable_embedding = embeddings
        
    clustering = AgglomerativeClustering(n_clusters=n)
    labels = clustering.fit_predict(clusterable_embedding)

    return labels, clusterable_embedding

def visualize_clusters(clusterable_embedding, labels, method="UMAP"):
    """Visualizes clusters using given method of dimensionality reduction to 2d."""
    logging.info("Running visualization...")
    if method=="TSNE":
        tsne_coords = reduce(clusterable_embedding, 2, method="TSNE")

        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=labels, cmap='rainbow', alpha=0.6)
        plt.title("Sentence Clusters")
        plt.show()
    else:
        umap_2d = reduce(clusterable_embedding, 2, method="UMAP")
        plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, cmap='rainbow', alpha=0.6, s=2)
        plt.title("Sentence Clusters (UMAP Projection)")
        plt.show()
