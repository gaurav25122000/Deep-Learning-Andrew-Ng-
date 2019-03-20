import numpy as np
def initialize_params_zero(layer_size):
    params = {}
    L = len(layer_size)
    for i in range(1,L):
        params['W'+str(i)] = np.zeros(shape = (layer_size[i],layer_size[i-1]))
        params['b'+str(i)] = np.zeros(shape = (layer_size[i],1))

    return params

def initialize_params_random(layer_size):
    L = len(layer_size)
    params = {}
    for i in range(1,L):
        params['W'+str(i)] = np.random.randn(layer_size[i],layer_size[i-1])
        params['b'+str(i)] = np.zeros((layer_size[i],1))
    return params

def initialize_params_he(layer_size):
    L = len(layer_size)
    params = {}
    for i in range(1,L):
        params['W'+str(i)] = np.random.randn(layer_size[i],layer_size[i-1])*(np.sqrt(2/layer_size[i-1]))
        params['b'+str(i)] = np.zeros((layer_size[i],1))
    return params
