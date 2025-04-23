# uses ANN and reflects real life Recall Stage for k most similar users and their last N items interaction 
import pandas as pd
import numpy as np
import faiss
import time
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

# Parameters
K = 5         # number of neighbors
N = 10        # last-N recent items per neighbor
TOP_X = 5     # top recommendations to return
nlist = 25   # number of coarse clusters for IVF
nprobe = 5   # number of clusters to search

file_path = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

# Load and process data
print("[INFO] Loading interaction data...")
interactions = pd.read_csv(file_path, sep="\t", header=None,
                           names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
                           on_bad_lines='skip')

print("[INFO] Converting timestamps and sorting interactions...")
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
interactions = interactions.sort_values(by=["user_id", "timestamp"])

# Map IDs to indices
print("[INFO] Mapping user and artist IDs to indices...")
unique_users = interactions["user_id"].unique()
unique_artists = interactions["artist_name"].unique()
user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
artist_name_to_idx = {aid: idx for idx, aid in enumerate(unique_artists)}
user_ids = np.array(unique_users)
artist_ids = np.array(unique_artists)

# Build dense user-artist matrix
print("[INFO] Creating user-artist matrix...")
user_artist_counts = interactions.groupby(["user_id", "artist_name"]).size().reset_index(name="playcount")
n_users = len(user_id_to_idx)
n_artists = len(artist_name_to_idx)
user_matrix = np.zeros((n_users, n_artists), dtype='float32')

for row in user_artist_counts.itertuples(index=False):
    user_idx = user_id_to_idx[row.user_id]
    artist_idx = artist_name_to_idx[row.artist_name]
    user_matrix[user_idx, artist_idx] += row.playcount

print("[INFO] Normalizing user vectors for cosine similarity...")
faiss.normalize_L2(user_matrix)

# FAISS index with IVF
print("[INFO] Building FAISS IndexIVFFlat...")
d = user_matrix.shape[1]
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

print("[INFO] Training FAISS index...")
index.train(user_matrix)
index.add(user_matrix)
index.nprobe = nprobe

print("[INFO] Searching nearest neighbors...")
start_faiss = time.time()
distances, neighbors = index.search(user_matrix, K + 1)
print(f"[TIMER] FAISS search completed in {time.time() - start_faiss:.2f}s")

# Build known and recent frequency maps
print("[INFO] Building known artist sets and recent artist frequencies...")
interactions["artist_idx"] = interactions["artist_name"].map(artist_name_to_idx)
user_to_known_artists = (
    interactions.groupby("user_id")["artist_idx"]
    .agg(set)
    .apply(lambda s: set(s))
    .to_dict()
)
# map keys to user indices
user_to_known_artists = {
    user_id_to_idx[uid]: artist_set for uid, artist_set in user_to_known_artists.items()
}
# Ensure interactions are sorted!
interactions = interactions.sort_values(by=["user_id", "timestamp"])

# Get last-N rows per user (faster than apply-tail)
last_n = interactions.groupby("user_id").tail(N)

# Count frequencies directly
recent_freq_df = last_n.groupby(["user_id", "artist_idx"]).size().reset_index(name="freq")

# Now convert to nested dict: {user_idx: {artist_idx: freq, ...}}
user_recent_frequencies_idx = defaultdict(dict)
for row in recent_freq_df.itertuples(index=False):
    u_idx = user_id_to_idx[row.user_id]
    user_recent_frequencies_idx[u_idx][row.artist_idx] = row.freq

# Optimized recommendation function
def recommend_for_user(u):
    seed_user_id = user_ids[u]
    seed_user_known = user_to_known_artists.get(u, set())
    neighbor_idxs = neighbors[u][1:]  # skip self
    neighbor_sims = distances[u][1:]

    candidate_scores = np.zeros(n_artists, dtype=np.float32)

    for i, neighbor_idx in enumerate(neighbor_idxs):
        sim_score = neighbor_sims[i]
        for artist, freq in user_recent_frequencies_idx.get(neighbor_idx, {}).items():
            candidate_scores[artist] += sim_score * freq

    # Mask known artists
    known_artists = list(seed_user_known)
    candidate_scores[known_artists] = 0

    top_artist_indices = candidate_scores.argsort()[-TOP_X:][::-1]

    return {
        "user_id": seed_user_id,
        "recommended_artists": top_artist_indices,
        "recommended_names": [artist_ids[i] for i in top_artist_indices]
    }

# Run in parallel with progress bar
print("[INFO] Starting parallel recommendation generation...")
start_time = time.time()
results = Parallel(n_jobs=-1)(
    delayed(recommend_for_user)(u) for u in tqdm(range(n_users), desc="Recommending")
)
end_time = time.time()

# Show sample results
for res in results[:5]:
    print("Seed user:", res["user_id"])
    print("Recommended artists:", res["recommended_names"])
    print("-" * 50)

print(f"[TIMER] Average time per user: {(end_time - start_time) / n_users:.4f}s")
print("[INFO] Done.")

# Output:
# [INFO] Loading interaction data...
# [INFO] Converting timestamps and sorting interactions...
# [INFO] Mapping user and artist IDs to indices...
# [INFO] Creating user-artist matrix...
# [INFO] Normalizing user vectors for cosine similarity...
# [INFO] Building FAISS IndexIVFFlat...
# [INFO] Training FAISS index...
# [INFO] Searching nearest neighbors...
# [TIMER] FAISS search completed in 4.01s
# [INFO] Building known artist sets and recent artist frequencies...
# [INFO] Starting parallel recommendation generation...
# Recommending: 100%|███████████████████████████████████████████████████| 992/992 [01:01<00:00, 16.19it/s]
# Seed user: user_000001
# Recommended artists: ['Mokira', 'Moderat', 'Pixies', 'Cristina Donà', 'Morgan']
# --------------------------------------------------
# Seed user: user_000002
# Recommended artists: ['My Morning Jacket', 'Conor Oberst With Gillian Welch', 'Kevin Drew', 'Sharon Jones And The Dap-Kings', 'The New Pornographers']
# --------------------------------------------------
# Seed user: user_000003
# Recommended artists: ['Bombers', 'Starsailor', 'Jamie Foxx', 'Scatman John', 'Laurent Wolf']
# --------------------------------------------------
# Seed user: user_000004
# Recommended artists: ['Dream 2 Science', 'Fake?', 'E.S. Posthumus', 'Foreign Born', 'Ill Bill']
# --------------------------------------------------
# Seed user: user_000005
# Recommended artists: ['Lax Project', 'De-Phazz', 'Toshack Highway', 'The Knife', 'Ferry Corsten']
# --------------------------------------------------
# [TIMER] Average time per user: 0.0627s
# [INFO] Done.