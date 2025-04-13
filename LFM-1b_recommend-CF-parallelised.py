import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import random
import time


K = 5

file_path = r'C:\MRS\lastfm-dataset-1K\lastfm-dataset-1K\userid-timestamp-artid-artname-traid-traname.tsv'
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

import concurrent.futures

def recommend_for_user(u, UAM, K):
    pc_vec = UAM.getrow(u)
    uU_sim = pc_vec.dot(UAM.transpose()).tocoo()

    uU_data, uU_user_idx = uU_sim.data, uU_sim.col
    uU_data[uU_user_idx == u] = 0

    uU_sim.data = uU_data
    uU_sim.eliminate_zeros()

    uU_user_idx, uU_data = uU_sim.col, uU_sim.data

    sort_index = np.argsort(uU_data)
    recommended_user_idx = uU_user_idx[sort_index[-K:]]
    recommended_user_ids = user_ids[recommended_user_idx]

    recommended_artists_idx = []
    for u_idx in recommended_user_idx:
        recommended_artists_idx.extend(UAM.getrow(u_idx).indices)

    recommended_artists_idx = sorted(set(recommended_artists_idx))
    recommended_artists_idx = np.setdiff1d(recommended_artists_idx, pc_vec.indices)

    if len(recommended_artists_idx) >= 5:
        random_indices = random.sample(list(recommended_artists_idx), 5)
    else:
        random_indices = list(recommended_artists_idx)

    return {
        'seed_user': user_ids[u],
        'neighbors': recommended_user_ids,
        'recommended_artist_indices': random_indices,
        'recommended_artist_names': [artist_ids[i] for i in random_indices]
    }

if __name__ == '__main__':
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda u: recommend_for_user(u, UAM, K), range(UAM.shape[0])))

    for res in results:
        print("Seed user-id:", res['seed_user'])
        print("Nearest K={} neighbors' user-ids:".format(K), res['neighbors'])
        print("Indices of recommended artists:", res['recommended_artist_indices'])
        print("Recommended artist names:", res['recommended_artist_names'])
        print('-' * 80)

    end_time = time.time()
    total_users = UAM.shape[0]
    avg_time = (end_time - start_time) / total_users
    print(f"Average time per user recommendation: {avg_time:.4f} seconds")

