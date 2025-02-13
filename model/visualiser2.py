import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('outputs/embeddings.csv')

     # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(data['reduced_x'], data['reduced_y'], alpha=0.7, edgecolor='w', s=100)
    coords = data[["reduced_x", "reduced_y"]].to_numpy( )
    # Add filename labels
    for i, name in enumerate(data["filename"]):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), 
                     textcoords="offset points", 
                     xytext=(0, 5), 
                     ha='center', 
                     fontsize=8)

    plt.title('Document Embeddings Visualization (TSNE)', pad=20)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

if (__name__ == "__main__"):
    main()
