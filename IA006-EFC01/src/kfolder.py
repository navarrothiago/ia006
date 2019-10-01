'''
Created on 21 de set de 2019

@author: navarro
'''

import numpy as np

class Kfolder(object):
    '''
    classdocs
    '''

    def __init__(self, split):
        '''
        Constructor
        '''
        self.split = split
        self.pointer = 0
    
    def split_data (self, data):
        train = np.array([])
        test = np.array([])
        data = np.array(data)
        block = 0
        
        if((len(data) % self.split) == 0):
            block = int(len(data)/self.split)
            test = data[self.pointer : self.pointer + block]
            self.pointer = (self.pointer + block) % len(data)
            
            for i in range(self.split - 1):
                # Merge das duas listas
                train = np.concatenate([train, data[self.pointer : self.pointer + block]])
                self.pointer = (self.pointer + block) % len(data)
        
        self.pointer = (self.pointer + block) % len(data)
        return train.reshape(len(train),1), test.reshape(len(test),1)