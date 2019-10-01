'''
Created on 21 de set de 2019

@author: navarro
'''

import numpy as np

# phi(a_y(a_n)) = [ 1 a_y(a_n-1) ... a_y(a_n-K) ]^T
def phi(a_y, n, k):
    phi1 = np.ones((1,1))
    x_r = a_y[n-k:n]
    phi2 = np.array(x_r[::-1])
    return np.concatenate([phi1, phi2])