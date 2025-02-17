import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rag(query, embeddings, top_n=3, threshold=0.5):
    """
    Extracts top keywords from a text segment using KeyBERT.
    """
    
    # Segment text into sentences
    query = nltk.sent_tokenize(query)
    if not query:
        logging.warning("No sentences detected in document.")
        return []

    # Initialize models
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate sentence embeddings
    query = model.encode(query)
    results = [(0, -1)] * top_n
    for i, e in enumerate(embeddings):
        dot = np.dot(query, e)
        for j in range(results):
            if dot > results[j][1]:
                results[j] = (i, dot)

    return [x[0] for x in results if x[1] > threshold]

