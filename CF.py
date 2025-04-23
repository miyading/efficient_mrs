import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import random
import time 

K = 5

file_path = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
# Load interaction log
interactions = pd.read_csv(file_path, sep="\t", header=None,
                           names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"], on_bad_lines='skip')

# Count number of times each user listened to each artist
playcounts = interactions.groupby(["user_id", "artist_name"]).size().reset_index(name="playcount")

# Create mappings: user/artist to index
user_id_to_idx = {user: idx for idx, user in enumerate(playcounts["user_id"].unique())}
artist_name_to_idx = {artist: idx for idx, artist in enumerate(playcounts["artist_name"].unique())}

# Map to index
playcounts["user_idx"] = playcounts["user_id"].map(user_id_to_idx)
playcounts["artist_idx"] = playcounts["artist_name"].map(artist_name_to_idx)

# Build the sparse matrix (UAM)
UAM = csr_matrix((playcounts["playcount"],
                  (playcounts["user_idx"], playcounts["artist_idx"])),
                 shape=(len(user_id_to_idx), len(artist_name_to_idx)))

# normalize rows (for cosine similarity)
from sklearn.preprocessing import normalize
UAM = normalize(UAM, norm='l2', axis=1)

# Create lists for reverse lookup
user_ids = np.array(list(user_id_to_idx.keys()))
artist_ids = np.array(list(artist_name_to_idx.keys()))


if __name__ == '__main__':
    start_time = time.time()
    for u in range(0, UAM.shape[0]):
        print("Seed user-id: " + str(user_ids[u]))

        # Get normalized playcount vector for current user
        pc_vec = UAM.getrow(u)

        # Compute cosine similarity between this user and all users
        uU_sim = pc_vec.dot(UAM.transpose()).tocoo()
        uU_user_idx = uU_sim.col
        uU_data = uU_sim.data

        # Remove self-similarity
        uU_data[uU_user_idx == u] = 0

        # Eliminate zeros
        uU_sim.data = uU_data
        uU_sim = uU_sim.tocsr()
        uU_sim.eliminate_zeros()
        uU_sim = uU_sim.tocoo()

        # Re-assign user indices and scores
        uU_user_idx = uU_sim.col
        uU_data = uU_sim.data

        # Sort by similarity score
        sort_index = np.argsort(uU_data)

        # Select top-K nearest neighbors
        # Note that uU_user_idx indeed provides the indices for users in UAM
        recommended_user_idx = uU_user_idx[sort_index[-K:]]
        # Get user_ids corresponding to nearest neighbors
        recommended_user_ids = user_ids[recommended_user_idx]
        # Get similarity score for nearest neighbors
        recommended_user_scores = uU_data[sort_index[-K:]]

        print("Nearest K=" + str(K) + " neighbors' user-ids: ", recommended_user_ids.flatten())

        # Get all artists these similar users have listened to
        recommended_artists_idx = []
        for u_idx in recommended_user_idx:
            recommended_artists_idx.extend(list(UAM.getrow(u_idx).indices))

        # Remove duplicates and sort
        recommended_artists_idx = sorted(set(recommended_artists_idx))

        # Remove artists already known to seed user
        recommended_artists_idx = np.setdiff1d(recommended_artists_idx, pc_vec.indices)

        # Narrow down to random 5 artist out of all artists that similar users listen to for illustration purpose
        random_indices = random.sample(range(len(recommended_artists_idx)), 5)
        random_elements = [recommended_artists_idx[i] for i in random_indices]

        print("Indices of " + str(len(random_elements)) + " recommended artists: ", random_elements)
        print("Recommended artist names:", [artist_ids[i] for i in random_elements])
        print('-' * 80)

    end_time = time.time()
    total_users = UAM.shape[0]
    avg_time = (end_time - start_time) / total_users
    print(f"Average time per user recommendation: {avg_time:.4f} seconds")