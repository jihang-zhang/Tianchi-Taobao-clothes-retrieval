import numpy as np

def normalize(v):
    norm = np.sqrt((v**2).sum())
    if norm == 0: 
       return v
    return v / norm