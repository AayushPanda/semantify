import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

def visualise_embeddings(embeddings_csv, output_file='embedding_plot.png'):
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

def visualise_embeddings_3d(embeddings_csv, output_file='embedding_plot.png'):
    """Main visualization function"""
    # Load data
    df = pd.read_csv(embeddings_csv)
    filenames = df['filename']
    embeddings = df.drop('filename', axis=1).values

    # Compute distance matrix
    distances = compute_pairwise_distances(embeddings)

    # Compute MDS coordinates
    coords_3d = classical_mds(distances, n_components=3)

    # Create plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], alpha=0.7, edgecolor='w', s=100)

    # Add filename labels
    for i, name in enumerate(filenames):
        ax.text(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], name)

    ax.set_title('Document Embeddings Visualization (MDS)', pad=20)
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_zlabel('MDS Dimension 3')
    ax.grid(alpha=0.2)
    # ax.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualise document embeddings in 2D')
    parser.add_argument('embeddings_csv', help='Input CSV file with embeddings')
    parser.add_argument('-o', '--output', default='embedding_plot.png', help='Output image filename')
    parser.add_argument('--d3', action='store_true', help='Visualise in 3D (buggy)')
    args = parser.parse_args()
    if args.d3:
        visualise_embeddings_3d(args.embeddings_csv, args.output)
    else:
        visualise_embeddings(args.embeddings_csv, args.output)