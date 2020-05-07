import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances


def save_data_with_pickle(relative_path, data):
    """ Save data using pickle (serialize) """

    with open(relative_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_data_with_pickle(relative_path):
    """ Load data using pickle (deserialize) """

    with open(relative_path, 'rb') as handle:
        return pickle.load(handle)


def hyper_distance(tensor1, tensor2):
    return np.arccosh(1 + ((2 * (np.linalg.norm(tensor1 - tensor2) ** 2)) / ((1 - np.linalg.norm(tensor1) ** 2)*(1 - np.linalg.norm(tensor2) ** 2))))

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def cosine_dissimilarity(a, b):
    return 1 - cosine_similarity(a, b)

def vector_mean(vectors):
    return np.mean(vectors, axis= 0)

def hyperbolic_midpoint(vectors, r = 1):

    hv = [hyperboloid_projection(vs, r) for vs in vectors]
    summ = sum(hv)
    midpoint = summ/np.sqrt(lorentzian(summ, summ))
    poincare_midpoint = inverse_projection(midpoint, r)
    return poincare_midpoint 


def lorentzian(v0, v1):
    prod = 0
    for i in range(len(v0) - 1):
        prod += v0[i] * v1[i]
    
    ret = - prod + (v0[len(v0) - 1] * v1[len(v1) - 1])
    return abs(ret)


def hyperboloid_projection(v, r):
    n = norm(v)
    t = [(r**2 + (n ** 2)) / (r**2 - (n ** 2))]
    projected = [(2 * r**2 * vs) /(r**2 - (n ** 2)) for vs in v]
    projected.extend(t)
    return np.array(projected)

def inverse_projection(v, r):
    return np.array([vs/(r**2 + v[-1]) for vs in v[:-1]])


def euclidean_similarity(*args):
    return 1/(1 + euclidean_distances(*args))


LOSSES = {'cosine_dissimilarity': 'COSD',
            'hyperbolic_distance': 'HYPD',
            'normalized_hyperbolic_distance': 'NHYPD',
            'regularized_hyperbolic_distance': 'RHYPD',
            'hyperboloid_distance' : 'LORENTZD',
            'multilabel_Minimum_Normalized_Poincare': 'NHMML',
            'multilabel_Minimum_Poincare': 'HMML',
            'multilabel_Minimum_Cosine': 'DMML',
            'multilabel_Average_Poincare': 'HMAL',
            'multilabel_Average_Cosine': 'DMAL', 
        }