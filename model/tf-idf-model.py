import os
import re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE   

# unlimited.. poWAAHHH :D
import resource
resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


try:
    import nltk
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
except ImportError:
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    print("NLTK not found, using fallback stopwords list")
 
nltk.download('stopwords')

def read_text_files(directory):
    """Read all .txt files from a directory into a DataFrame"""
    filenames, texts = [], []
    for fn in os.listdir(directory):
        if fn.endswith('.txt'):
            with open(os.path.join(directory, fn), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            filenames.append(fn)
    return pd.DataFrame({'filename': filenames, 'text': texts})

def preprocess(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()

    return [word for word in tokens if word not in stopwords]

def build_vocab(tokenised_docs, min_df=2):
    """Build vocabulary with minimum document frequency filtering"""
    word_counts = {}
    for doc in tokenised_docs:
        for word in set(doc):
            word_counts[word] = word_counts.get(word, 0) + 1
    return [word for word, count in word_counts.items() if count >= min_df]

# https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
# https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
import numpy as np

def tfidf_transform(tokenised_docs, vocab, alpha=2.0, entropy_weighting=True):
    """Compute TF-IDF matrix with optional scaling and information theoretic reweighting"""
    word2idx = {word: i for i, word in enumerate(vocab)}
    n_docs = len(tokenised_docs)
    n_vocab = len(vocab)
        
    # Term Frequency (TF)
    tf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(tokenised_docs):
        for word in doc:
            if word in word2idx:
                tf[i, word2idx[word]] += 1
        if len(doc) > 0:
            tf[i] /= np.sum(tf[i])  # L1 Normalization
    
    # Document Frequency (DF)
    df = np.sum(tf > 0, axis=0)
    
    # IDF scaling
    idf = np.log((tf + 1) / (df + 1)) ** alpha + 1  # Exponentiate IDF for stronger weighting
    
    # Optional: Entropy-based reweighting to ignore low information words
    if entropy_weighting:
        p_w = df / np.sum(df)  # Probability of word occurrence across documents
        entropy = -p_w * np.log2(p_w + 1e-9)  # Entropy of word distribution
        idf *= (1 + entropy)  # Boost less uniformly distributed words
    
    return tf * idf


# TODO: try doing the force directed graph we came up with in franks bedroom :tongue emoji:
def pca_embedding(X, n_components=100):
    """Dimensionality reduction using PCA"""
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Efficient covariance calculation
    cov = np.cov(X_centered, rowvar=False)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues descending
    sorted_idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, sorted_idx[:n_components]]
    
    # Project data
    return np.dot(X_centered, components)

def main(input_dir, output_file='embeddings.csv', embed_dim=50):
    # Read and preprocess documents
    df = read_text_files(input_dir)
    df['tokens'] = df['text'].apply(preprocess)
    
    # Build vocabulary with minimum document frequency
    vocab = build_vocab(df['tokens'])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Compute TF-IDF matrix
    print("Computing TF-IDF matrix...")
    tfidf_matrix = tfidf_transform(df['tokens'], vocab)
    
    # Compute PCA embeddings
    print(f"Computing PCA embeddings (dim={embed_dim})...")
    embeddings = pca_embedding(tfidf_matrix, embed_dim)
    
    # Create output DataFrame
    print("Creating output DataFrame...")
    embedding_cols = [f'emb_{i}' for i in range(embed_dim)]
    result_df = pd.DataFrame(embeddings, columns=embedding_cols)
    result_df.insert(0, 'filename', df['filename'])

    # Create estimator
    print("Creating TSNE estimator...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_coords = tsne.fit_transform(embeddings)
    result_df['reduced_x'] = tsne_coords[:, 0]
    result_df['reduced_y'] = tsne_coords[:, 1]
    
    # Save to CSV
    print(f"Saving embeddings to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate document embeddings from text files')
    parser.add_argument('input_dir', help='Directory containing text files')
    parser.add_argument('-o', '--output', default='embeddings.csv', help='Output CSV filename')
    parser.add_argument('-d', '--dim', type=int, default=100, help='Embedding dimension')
    args = parser.parse_args()
    
    main(args.input_dir, args.output, args.dim)