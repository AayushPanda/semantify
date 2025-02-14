import segmenter
import clustering
import pandas as pd
import os
import logging

def filesEmbedder(path, n_important):
    """Reads all files from a directory and creates embeddings for them absed on the n most important semantic segments"""
    embeddings = pd.DataFrame(columns=["file", "embedding"])
    paths = os.listdir(path)
    lpaths = len(paths)
    c = 0
    for file in paths:
        logging.info(f"Progress: {c}/{lpaths}")
        raw_embeds = [res["embedding"][:n_important] for res in sorted(segmenter.process_document(path + file), key=lambda x: x["num_sentences"])]
        for emb in raw_embeds:
            embeddings.loc[len(embeddings)] = [file, emb]

    return embeddings

out = filesEmbedder("data/docs/", 3)
clustering.visualize_clusters(out["embedding"], out["file"])
