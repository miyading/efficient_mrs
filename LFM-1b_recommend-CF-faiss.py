# updated script using FAISS for user-user similarity instead of computing dense dot products with sparse matrices
# uses FAISS.IndexFlatIP to compute cosine sim; index.search() for fast ANN
import pandas as pd
import numpy as np
import random
import time
import faiss

K = 5

# File path
file_path = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

# Load interaction log
interactions = pd.read_csv(file_path, sep="\t", header=None,
                           names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
                           on_bad_lines='skip')

# Count number of times each user listened to each artist
playcounts = interactions.groupby(["user_id", "artist_name"]).size().reset_index(name="playcount")

# Create mappings: user/artist to index
user_id_to_idx = {user: idx for idx, user in enumerate(playcounts["user_id"].unique())}
artist_name_to_idx = {artist: idx for idx, artist in enumerate(playcounts["artist_name"].unique())}

# Map to index
playcounts["user_idx"] = playcounts["user_id"].map(user_id_to_idx)
playcounts["artist_idx"] = playcounts["artist_name"].map(artist_name_to_idx)

# Build dense user-artist matrix (playcounts)
n_users = len(user_id_to_idx)
n_artists = len(artist_name_to_idx)
user_matrix = np.zeros((n_users, n_artists), dtype='float32')

for row in playcounts.itertuples(index=False):
    user_matrix[row.user_idx, row.artist_idx] += row.playcount

# Normalize rows to unit length for cosine similarity
faiss.normalize_L2(user_matrix)

# Create FAISS index for cosine similarity (inner product after normalization)
index = faiss.IndexFlatIP(n_artists)
index.add(user_matrix)  # Add all user vectors

# Reverse lookup
user_ids = np.array(list(user_id_to_idx.keys()))
artist_ids = np.array(list(artist_name_to_idx.keys()))

if __name__ == '__main__':
    start_time = time.time()

    # Search all users against the index
    distances, neighbors = index.search(user_matrix, K + 1)  # +1 includes self

    for u in range(n_users):
        print("Seed user-id:", user_ids[u])

        # Remove self (first neighbor)
        neighbor_indices = neighbors[u][1:]  # skip self
        neighbor_scores = distances[u][1:]

        recommended_artists_idx = []
        for neighbor in neighbor_indices:
            neighbor_vector = user_matrix[neighbor]
            recommended_artists_idx.extend(np.where(neighbor_vector > 0)[0])

        # Remove artists already known to the user
        known_artists = np.where(user_matrix[u] > 0)[0]
        recommended_artists_idx = list(set(recommended_artists_idx) - set(known_artists))

        if len(recommended_artists_idx) >= 5:
            random_indices = random.sample(recommended_artists_idx, 5)
        else:
            random_indices = recommended_artists_idx  # fallback

        print("Nearest K={} neighbors' user-ids:".format(K), user_ids[neighbor_indices])
        print("Indices of {} recommended artists:".format(len(random_indices)), random_indices)
        print("Recommended artist names:", [artist_ids[i] for i in random_indices])
        print('-' * 80)

    end_time = time.time()
    avg_time = (end_time - start_time) / n_users
    print(f"Average time per user recommendation: {avg_time:.4f} seconds")