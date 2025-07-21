import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_largest_cluster(sentences: list[str]):
    """
    Computes the largest cluster of sentences using K-means clustering, finds the sentences within the largest cluster, and orders them by their distance to the cluster center.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, ordered by their proximity
              to the cluster center.
    """
    if len(sentences) == 0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]

    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster


def compute_kmeans(sentences: list[str]):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings, normalizing the embeddings, and determining the optimal number of clusters using binary search.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - kmeans (KMeans): The KMeans object with the optimal number of clusters determined.
    """
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences)
    # normalize the embeddings for equivalent computation of the cosine distance
    embeddings = preprocessing.normalize(embeddings)
    # compute the number of clusters with binary search
    kmeans = binary_search_optimal_kmeans(embeddings, min_k=0, max_k=(len(sentences) - 1))
    return embeddings, kmeans


def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k for KMeans clustering using binary search on the silhouette score.

    Args:
        data (list): cluster data.
        min_k: minimum k for binary search
        max_k: maximum k for binary search

    Returns:
        list: List of cleaned response strings.
    """
    best_k = min_k
    best_score = -1
    best_kmeans = KMeans(n_clusters=1, random_state=42).fit(
        data
    )  # start with 1 cluster for len(data) < 2

    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:
            break

        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)

        if score > best_score:
            best_score = score
            best_k = mid_k
            best_kmeans = kmeans  # Update the best KMeans model
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    return best_kmeans
