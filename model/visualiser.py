import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_pairwise_distances(embeddings):
    """Compute pairwise Euclidean distances using vectorized operations"""
    squared_diff = np.sum((embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]) ** 2, axis=2)
    return np.sqrt(squared_diff)

def classical_mds(distances, n_components=2):
    """Classical MDS implementation using numpy"""
    n = distances.shape[0]
    
    # Squared dist matrix
    D_sq = distances ** 2

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Double centering
    B = -0.5 * H @ D_sq @ H

    # Ensure symmetry
    B = 0.5 * (B + B.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues, vectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Select top positive eigenvalues
    positive_mask = eigenvalues > 1e-8  # Account for floating point fuckery
    eigenvalues = eigenvalues[positive_mask]
    eigenvectors = eigenvectors[:, positive_mask]

    # Take top components
    eigenvectors = eigenvectors[:, :n_components]
    eigenvalues = eigenvalues[:n_components]

    # Project to smaller dimension
    return eigenvectors * np.sqrt(eigenvalues)

def visualize_embeddings(embeddings_csv, output_file='embedding_plot.png'):
    """Main visualization function"""
    # Load data
    df = pd.read_csv(embeddings_csv)
    filenames = df['filename']
    embeddings = df.drop('filename', axis=1).values

    # Compute distance matrix
    distances = compute_pairwise_distances(embeddings)

    # Compute MDS coordinates
    coords_2d = classical_mds(distances)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, edgecolor='w', s=100)

    # Add filename labels
    for i, name in enumerate(filenames):
        plt.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]), 
                     textcoords="offset points", 
                     xytext=(0, 5), 
                     ha='center', 
                     fontsize=8)

    plt.title('Document Embeddings Visualization (MDS)', pad=20)
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize document embeddings in 2D')
    parser.add_argument('embeddings_csv', help='Input CSV file with embeddings')
    parser.add_argument('-o', '--output', default='embedding_plot.png', help='Output image filename')
    args = parser.parse_args()
    
    visualize_embeddings(args.embeddings_csv, args.output)