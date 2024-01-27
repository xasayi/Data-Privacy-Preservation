import sys
import torch
import numpy as np
from numpy.linalg import norm
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

device = torch.device("mps")

def cosine(a, b):
    angle = np.dot(a.detach().cpu().numpy(), b.detach().cpu().numpy())/ (norm(a.detach().cpu().numpy())*norm(b.detach().cpu().numpy()))
    return angle

def euclidean(a, b):
    return np.sqrt(2-2*cosine(a.detach().cpu().numpy(), b.detach().cpu().numpy()))

def find_similar(pri_batch, pub_pool, function):
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

if __name__ == '__main__':
    a = np.array([1, 2])
    b = np.array([3, 2])
    print(cosine(a, b))
    print(euclidean(a, b))
    pri_batch = np.array([[1,2,3],[1,2,2], [3,2,1]])
    pub_pool = np.array([[1,2,3],[2,3,4],[1,2,8],[7,2,3],[2,5,4],[0,2,8]])
    indices = find_similar(pri_batch, pub_pool, 'euclid')
    print(indices)