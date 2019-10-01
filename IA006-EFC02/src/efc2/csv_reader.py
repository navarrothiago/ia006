'''
Created on 28 de set de 2019

@author: navarro
'''

import csv
import numpy as np
import pandas as pd

def load_csv():
    # carrega arquivo csv
    return pd.read_csv('dados_voz_genero.csv')

if __name__ == '__main__':
    load_csv()
