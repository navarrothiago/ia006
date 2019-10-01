'''
Created on 21 de set de 2019

@author: navarro
'''
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())