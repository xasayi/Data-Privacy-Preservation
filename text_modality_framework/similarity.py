'''
Function to find similar public samples for each private sample in a batch
'''
import torch
import numpy as np
from numpy.linalg import norm

device = torch.device("mps")

def cosine(a, b):
    '''cosine similarity'''
    a_norm = norm(a.detach().cpu().numpy())
    b_norm = norm(b.detach().cpu().numpy())
    angle = np.dot(a.detach().cpu().numpy(),b.detach().cpu().numpy())/(a_norm*b_norm)
    return angle

def euclidean(a, b):
    '''euclidean'''
    return np.sqrt(a.detach().cpu().numpy() - b.detach().cpu().numpy())

def find_similar(pri_batch, pub_pool, function):
    '''find similar public samples from a public pool for each sample in a private batch'''
    if function == 'cosine':
        f = cosine
    if function == 'euclid':
        f = euclidean
    indices = []
    for pri in pri_batch:
        similarities = []
        for pub in pub_pool:
            similarities.append(f(pri, pub))
        index = np.argmax(similarities)
        while index in indices:
            similarities[index] = 0
            index = np.argmax(similarities)
        indices.append(index)
    return indices
