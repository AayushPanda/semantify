import umap
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def reduce(embeddings, target_dims, method="UMAP"):
    """Reduces dimensionality of embeddings using given method."""
    if method=="TSNE":
        tsne = TSNE(n_components=target_dims, perplexity=min(30, len(embeddings) - 1), n_iter=300)
        tsne_coords = tsne.fit_transform(embeddings)
        return tsne_coords
    else:
        umapped = umap.UMAP(n_components=target_dims, n_neighbors=30, min_dist=0.2).fit_transform(embeddings)
        return umapped

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
