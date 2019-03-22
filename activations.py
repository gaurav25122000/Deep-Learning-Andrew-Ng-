import numpy as np
def relu(Z):
    return np.maximum(0,Z)
def sigmoid(Z):
    sig = 1/(1+np.exp(-Z))
    return sig
