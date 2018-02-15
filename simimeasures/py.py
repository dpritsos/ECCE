# -*- coding: utf-8 -*-

import numpy as np


def cosine_sim(vector, centroid):

    # Convert Arrays and Matrices to Matrix Type for preventing unexpected error, such as...
    # ...returning vector instead of scalar
    vector = np.matrix(vector)
    centroid = np.matrix(centroid)

    return vector * np.transpose(centroid) / (np.linalg.norm(vector) * np.linalg.norm(centroid))


def cos_sim_sprs(vector, centroid):

    return cosine_similarity(vector.todense(), centroid)


# TO BE ImPLAMENTED WITH Cython
def minmax_sim(v1, v2):

    return np.sum(np.min(np.vstack((v1, v2)), axis=0)) / np.sum(np.max(np.vstack((v1, v2)), axis=0))


def jaccard_sim(v0, v1):

    v0 = np.where((v0 > 0), 1, 0)
    v1 = np.where((v0 > 0), 1, 0)

    return 1.0 - spd.jaccard(v0, v1)


def hamming_sim(vector, centroid):

    return 1.0 - spd.hamming(centroid, vector)


def correl_sim(vector, centroid):

    vector = vector[0]
    centroid = np.array(centroid)[0]

    vector_ = np.where(vector > 0, 0, 1)
    centroid_ = np.where(centroid > 0, 0, 1)

    s11 = np.dot(vector, centroid)
    s00 = np.dot(vector_, centroid_)
    s01 = np.dot(vector_, centroid)
    s10 = np.dot(vector, centroid_)

    denom = np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
    if denom == 0.0:
        denom = 1.0

    return (s11*s00 - s01*s10) / denom
