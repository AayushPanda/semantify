import os
import logging
import pandas as pd
import numpy as np
import segmenter
import matplotlib.pyplot as plt
from keybert import KeyBERT
import clustering
from sklearn.metrics import silhouette_score
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def find_optimal_clusters(embeddings, min_clusters=2, max_clusters=10):
    """
    Uses the Silhouette score to determine the optimal number of clusters.
    Ensures that n_clusters is at least 2 to avoid errors.
    """
    best_n = min_clusters
    best_score = -1

    # Ensure we don't request more clusters than samples
    max_clusters = min(max_clusters, len(embeddings) - 1)

    if max_clusters < 2:  # If we can't create at least 2 clusters, return 2
        return 2

    for n in range(min_clusters, max_clusters + 1):
        labels, _ = clustering.ncluster(embeddings, n=n)

        if len(set(labels)) > 1:  # Silhouette score requires >1 cluster
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_n = n

    return best_n

def hierarchical_cluster_and_label(embeddings_df, min_cluster_size=5, level=1, max_levels=100):
    """
    Recursively performs hierarchical clustering and assigns meaningful topic labels as folder names.
    """
    if len(embeddings_df) < min_cluster_size or level > max_levels:
        return embeddings_df  # Base case: Stop recursion

    embeddings = np.vstack(embeddings_df["embedding"].to_list())

    # Determine optimal number of clusters
    n_clusters = find_optimal_clusters(embeddings, min_clusters=2, max_clusters=min(10, len(embeddings_df)))
    
    labels, _ = clustering.ncluster(embeddings, n=n_clusters)
    embeddings_df["cluster"] = labels
    
    # Generate topic labels using KeyBERT
    kw_model = KeyBERT()
    cluster_topic_labels = {}

    for cluster in np.unique(labels):
        sub_df = embeddings_df[labels == cluster]
        combined_text = " ".join(sub_df["topics"])
        
        keywords = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=1)
        topic_label = "_".join([kw[0] for kw in keywords])
        cluster_topic_labels[cluster] = topic_label

    # Append new cluster topic labels to appropriate rows
    if not "cluster-path" in embeddings_df.columns:
        embeddings_df["cluster-path"] = ""
    
    embeddings_df["cluster-path"] = embeddings_df.apply(lambda x: os.path.join(x["cluster-path"], cluster_topic_labels[x["cluster"]]), axis=1)

    # Recursively cluster within each subgroup
    clustered_dfs = []
    for cluster in np.unique(labels):
        sub_df = embeddings_df[labels == cluster].copy()
        clustered_dfs.append(hierarchical_cluster_and_label(sub_df, min_cluster_size, level + 1, max_levels))

    return pd.concat(clustered_dfs, ignore_index=True).drop(columns=["cluster"])

def filesEmbedder(path, n_important=3):
    """
    Reads all files from a directory and creates embeddings based on 
    the n most important semantic segments.
    """
    embeddings = pd.DataFrame(columns=["file", "embedding", "topics"])
    paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]

    for i, file in enumerate(paths):
        logging.info(f"Processing file {i+1}/{len(paths)}: {file}")

        segments = segmenter.process_document(file)
        if not segments:
            logging.warning(f"Skipping {file}, no segments found.")
            continue

        # Select top N most important segments based on number of sentences
        selected_segments = sorted(segments, key=lambda x: x["num_sentences"], reverse=True)[:n_important]

        # Extract embeddings
        for seg in selected_segments:
            embeddings.loc[len(embeddings)] = [file, seg["embedding"], "_".join(seg["topics"])]

    return embeddings

def generate_folder_structure(embeddings_df, output_dir):
    """
    Generates a folder structure based on hierarchical cluster labels.
    """
    for _, row in embeddings_df.iterrows():
        cluster_path = os.path.join(output_dir, row["cluster-path"])
        os.makedirs(cluster_path, exist_ok=True)
        file_name = os.path.basename(row["file"])
        try:
            os.symlink(row["file"], os.path.join(cluster_path, file_name))
        except FileExistsError:
            logging.warning(f"File {file_name} already exists in {cluster_path}")

def main():
    parser = argparse.ArgumentParser(description='Embedding Mass-Meaning Aggregator')
    parser.add_argument('input_dir', type=str, help='Path with text files to be clustered')
    parser.add_argument('output_dir', type=str, help='Path to store clustered files')
    parser.add_argument("-embed_out_dir" , type=str, help="Path to store embeddings")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    logging.info("Extracting embeddings...")
    embeddings_df = filesEmbedder(input_dir)

    logging.info("Clustering embeddings hierarchically...")
    embeddings_df = hierarchical_cluster_and_label(embeddings_df)

    if embeddings_df is not None:
        logging.info("Generating folder structure...")
        generate_folder_structure(embeddings_df, output_dir)
        logging.info(f"Folder structure generated at {output_dir}")

        if args.embed_out_dir:
            logging.info(f"Saving embeddings to {args.embed_out_dir}")
            
            #reduce embeddings to 2D for visualization
            embeddings = np.vstack(embeddings_df["embedding"].to_list())
            clusterable_embeddings = clustering.reduce(embeddings, 2)
            embeddings_df["embedding"] = clusterable_embeddings.tolist()

            embeddings_df.to_json(os.path.join(args.embed_out_dir, "embeddings.json"), orient="records")
            embeddings_df.to_csv(os.path.join(args.embed_out_dir, "embeddings.csv"), index=False)

if __name__ == "__main__":
    main()