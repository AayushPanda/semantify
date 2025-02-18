import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Initialize models
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def rag(query, embeddings, top_n=3, threshold=None):
    """
    RAG with cosine
    """
    sentences = nltk.sent_tokenize(query)
    if not sentences:
        logging.warning("No sentences detected in query.")
        return []

    # idea: use keybert extracted keywords for retrieval to improve relevance of retrieval??
    query_embedding = MODEL.encode(sentences, normalize_embeddings=True)
    similarity_scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]  # Sort descending
    
    # dynamic thres -- might not work well
    if threshold is None:
        threshold = np.mean(similarity_scores) * 0.8  # 80% of mean similarity

    # Filter out bad matches
    filtered_indices = [idx for idx in top_indices if similarity_scores[idx] >= threshold]

    logging.info(f"Top matches: {filtered_indices}, Similarities: {similarity_scores[filtered_indices]}")
    
    return filtered_indices
