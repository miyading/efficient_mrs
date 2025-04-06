# Implementation of a very simple and much to improve user-based collaborative filtering (CF) recommender.
# Author: Markus Schedl

# Load required modules
import csv
import numpy as np
import h5py
from scipy import sparse


UAM_MATLAB_FILE = 'LFM-1b_LEs.mat'         # Matlab .mat file where the listening events are stored
ARTISTS_FILE = "LFM-1b_artists.txt"        # artist names for UAM
USERS_FILE = "LFM-1b_users.txt"            # user names for UAM
K = 3                                      # maximum number of seed's neighbors to select


# Read the user-artist-matrix and corresponding artist and user indices from Matlab file
def read_UAM(m_file):
    mf = h5py.File(m_file, 'r')
    user_ids = np.array(mf.get('idx_users')).astype(np.int64)
    artist_ids = np.array(mf.get('idx_artists')).astype(np.int64)
    # Load UAM
    UAM = sparse.csr_matrix((mf['/LEs/']["data"],
                             mf['/LEs/']["ir"],
                             mf['/LEs/']["jc"])).transpose()    #.tocoo().transpose()
    # user and artist indices to access UAM
    UAM_user_idx = UAM.indices #UAM.row -> for COO matrix
    UAM_artist_idx = UAM.indptr #UAM.col -> for COO matrix
    return UAM, UAM_user_idx, UAM_artist_idx, user_ids, artist_ids


# Function to read metadata (users or artists)
def read_from_file(filename, col):                  # col = column to read from file
    data = []
    with open(filename, 'r') as f:                  # open file for reading
        reader = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header
        for row in reader:
            item = row[col]
            data.append(item)
    f.close()
    return data


# Main program
if __name__ == '__main__':
    # Initialize variables
    artists = []            # artists
    users = []              # users

    # Read UAM
    UAM, UAM_user_idx, UAM_artist_idx, user_ids, artist_ids = read_UAM(UAM_MATLAB_FILE)
    print 'Users: ', len(user_ids)
    print 'Artists: ', len(artist_ids)

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE, 1)
    users = read_from_file(USERS_FILE, 0)

    # For all users
    for u in range(0, UAM.shape[0]):
        print "Seed user-id: " + str(users[u])

        # get (normalized) playcount vector for current user u
        pc_vec = UAM.getrow(u)

        # Compute similarities as dot product between playcount vector of user and all users via UAM (assuming that UAM is already normalized)
#        print uU_sim_users
        uU_sim = pc_vec.dot(UAM.transpose()).tocoo()
        uU_user_idx = uU_sim.col
        uU_data = uU_sim.data

        #
        # Determine nearest neighbors to seed based on uUM
        #

        # Find the occurrence of the seed user in uU_data cols
        # and set to 0 so that it is not selected as its own NN
        occ_user_idx = (uU_user_idx == u)
        uU_data[occ_user_idx] = 0

        # Eliminate zeros
        uU_sim.data = uU_data
        uU_sim = uU_sim.tocsr()
        uU_sim.eliminate_zeros()
        uU_sim = uU_sim.tocoo()
        uU_user_idx = uU_sim.col
        uU_data = uU_sim.data

        # Sort users according to the similarity (uU_data)
        sort_index = np.argsort(uU_data)

        # Select the K nearest neighbors among all users
        # Note that uU_user_idx indeed provides the indices for users in UAM
        recommended_user_idx = uU_user_idx[sort_index[-K:]]
        # Get user_ids corresponding to nearest neighbors
        recommended_user_ids = user_ids[recommended_user_idx]
        # Get similarity score for nearest neighbors
        recommended_user_scores = uU_data[sort_index[-K:]]

        print "Nearest K=" + str(K) + " neighbors\' user-ids: ", recommended_user_ids.flatten()
#        print 'Scores/similarities:  ' + str(recommended_user_scores)
#        print 'Index in UAM for recommended user-ids: ' + str(recommended_user_idx)

        #
        # Determine set of recommended artists
        #
        recommended_artists_idx = []
        for u_idx in recommended_user_idx:
            recommended_artists_idx.extend(list(UAM.getrow(u_idx).indices))

        # Convert to set to remove duplicates and sort it
        recommended_artists_idx = sorted(set(recommended_artists_idx))
        # Remove artists already known to seed user
        recommended_artists_idx = np.setdiff1d(recommended_artists_idx, pc_vec.indices)

        print "Indices of " + str(len(recommended_artists_idx)) + " recommended artists: ", recommended_artists_idx
