import argparse
import os
import logging
import nltk
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from clustering import cluster, visualize_clusters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def embed_sentences(sentences, model):
    """Generates embeddings for sentences using Sentence Transformer."""
    return model.encode(sentences)


def extract_keywords(segment_text, kw_model, top_n=3):
    """Extracts top keywords from a text segment using KeyBERT."""
    keywords = kw_model.extract_keywords(segment_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

def process_document(file_path, vis=False):
    """Processes a document: reads, embeds, clusters, extracts topics, and visualizes clusters."""
    logging.info(f"Processing document: {file_path}")

    # Read document
    text = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Download NLTK tokenizer if necessary
    nltk.download('punkt', quiet=True)

    # Segment text into sentences
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        logging.warning("No sentences detected in document.")
        return []

    # Initialize models
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=model)

    # Generate sentence embeddings
    embeddings = embed_sentences(sentences, model)

    # Perform clustering
    cluster_labels, clusterable_embedding = cluster(embeddings)

    # Visualize clusters
    if vis: visualize_clusters(clusterable_embedding, cluster_labels)

    # Group sentences by cluster labels
    segments = {}
    for idx, label in enumerate(cluster_labels):
        segments.setdefault(label, []).append(sentences[idx])

    # Extract topics for each segment
    results = []
    for i, (label, segment) in enumerate(segments.items()):
        segment_text = " ".join(segment)
        topics = extract_keywords(segment_text, kw_model)

        # Generate embedding for the segment
        segment_embedding = model.encode(segment_text)

        results.append({
            "segment_number": i + 1,
            "text": segment_text.strip(),
            "num_sentences": len(segment),
            "topics": topics,
            "embedding": segment_embedding.tolist()
        })

    return results

def main():
    parser = argparse.ArgumentParser(description='Semantic segmenter')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        logging.error(f"File not found: {args.file_path}")
        return

    results = process_document(args.file_path, vis=True)

    if args.verbose:
        for result in results:
            print(f"\nSegment {result['segment_number']}:")
            print(f"Topics: {', '.join(result['topics'])}")
            print(f"Text: {result['text'][:]}...")  # Show only first 200 characters
            print(f"Embedding dimensions: {len(result['embedding'])}")
            print("-" * 80)

if __name__ == "__main__":
    main()
