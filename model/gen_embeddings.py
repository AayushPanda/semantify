from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt')

# Load the text file and preprocess it
with open('data/docs/Australia.txt', 'r') as file:
    text = file.read()

# Split text into sentences (or paragraphs)
sentences = nltk.sent_tokenize(text)

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for each sentence
embeddings = model.encode(sentences)

# Perform clustering using KMeans
num_topics = 5  # Specify number of topics
kmeans = KMeans(n_clusters=num_topics)
kmeans.fit(embeddings)

# Get the cluster labels
labels = kmeans.labels_

# Visualize using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Plot the clusters
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
plt.title('Sentence Clustering (Topic Modeling)')
plt.show()

# Print out sentences per topic
for topic in range(num_topics):
    print(f"Topic {topic}:")
    for i, label in enumerate(labels):
        if label == topic:
            print(f" - {sentences[i]}")
