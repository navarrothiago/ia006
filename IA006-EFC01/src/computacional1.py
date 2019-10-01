'''
Created on 15 de set de 2019

@author: navarro
'''

# O sistema deve montar o vetor a_n(a_n) baseado nos paremetro K que representa
# as K ultimas amostras referente ao tempo a_n 

# O sistema deve montar a matrix phi baseado no valor de K

# phi(a_n(a_n)) = [ 1 a_y(a_n-1) ... a_y(a_n-K) ]^T, onde K são as K amostras passadas
# a_w = [ w0 w1 ... wK ]^T
# ^y(a_y(a_n)) = phi(a_y(a_n))^T a_w

# para K = 1, temos
# a_w = [w0 w1 ]
# phi_(a_y(a_n)) = [ 1 a_y(a_n-1) ]
# ^y(a_y(a_n)) = [ 1 a_y(a_n-1) ]^T . [w0 w1 ] 

# ^a_ye = [ ^y(a_y(0)) ^y(a_y(1)) ... ^y(a_y(N-1)) ]^T
# a_phi = [ phi_(a_y(0))^T phi_(a_y(1))^T ... phi_(a_y(N-1))^T ]^T 

import csv
import numpy as np
import numpy.linalg as npl
from datetime import datetime, date
import matplotlib.pyplot as plt


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
        
def load_array_n_x():
    a_n = []
    a_y = []

    with open('/home/navarro/eclipse-workspace/IA006-EFC01/src/daily-minimum-temperatures.csv') as csvfile:
        start_time = datetime.now()
        row_count = 0
        readCSV = csv.reader(csvfile, delimiter=',')
        i = 0   
        for row in readCSV:
            if(row[0] == "Date" and row[1] == "Temp"):
                i = i + 1
                continue
            date = datetime.strptime(row[0], '%Y-%m-%d')
            temperature = row[1]
            
            if(i == 1):
                start_time = date
                print(start_time)
        
            i = i + 1
            row_count
            delta_days = date - start_time
            a_n.append(delta_days.days)
            a_y.append(float(temperature))
    
        a_n = np.array([a_n]).T
        a_y = np.array([a_y]).T
        return a_n, a_y, i - 1, start_time

def init_array_w(k):
    return np.ones((k + 1, 1))

# def gradient(a_ye, a_ye, a_phi):
#     a_error = a_ye - a_ye
#     return -2 * a_error.T.dot(a_phi)

# phi(a_y(a_n)) = [ 1 a_y(a_n-1) ... a_y(a_n-K) ]^T
def phi(a_y, n, k):
    phi1 = np.ones((1,1))
    x_r = a_y[n-k:n]
    phi2 = np.array(x_r[::-1])
    return np.concatenate([phi1, phi2])

if __name__ == '__main__':
    print("Iniciando main")
    
    a_n, a_y, N, start_time = load_array_n_x()
    k_fold = 9
    a_phi = None
    a_k = range(1,31)
    a_rmse = np.zeros((k_fold, len(a_k) + 1, 2))
    
    block = int(len(a_y[:,0])/10)
    
    kf = Kfolder(k_fold)
    
    # Matriz (K  x 2) para guardar os da média RMSE 
    a_rmse_mean = np.zeros((31,2))
    
    #[kfold K rmse]
    min_rmse = np.full(3, np.inf)
    
    min_k_fold = None
    min_k = None

    # Para todo o k no range de 1 a 31
    for k in a_k:
        # Varre todas as pastas. 
        for fold in range(k_fold):
            train, test = kf.split_data(a_y[0: k_fold*block,0])
            
            a_ye = np.zeros((len(train), 1))
            print("Calculando para K = ", k, " e para fold = ", fold)
            a_w = init_array_w(k);
            a_phi = np.zeros((len(train), k + 1))
            
            # Cálculo do vetor phi baseado no dados de treinamento
            for i in range(k, len(train)):
                a_phi_x = np.array(phi(train, i, k))
                a_phi[i] = a_phi_x.T
    
            # Matriz phi.T . phi deve ser inversível
            assert(npl.matrix_rank(a_phi) == min(a_phi.shape[0], a_phi.shape[1]))
            
            # Solução fechada quando tem inversa
            a_w = npl.inv((a_phi.T).dot(a_phi)).dot(a_phi.T.dot(train))
                
            # Cálculo do erro em relação ao dados de validação (test)
            for i in range(k, len(test)):
                ye = np.dot(phi(test, i, k).T, a_w)
                a_ye[i] = np.reshape(ye,1)
            
            # Cálculo da RMSE para um k específico.
            a_rmse[fold, k, 0] = rmse(a_ye[k:len(test)], test[k:len(test)])
            
            
            # Cálculo do erro em relação ao dados de treinamento (test)
            for i in range(k, len(train)):
                ye = np.dot(phi(train, i, k).T, a_w)
                a_ye[i] = np.reshape(ye,1)
            
            # Cálculo da RMSE para um k específico.
            a_rmse[fold, k, 1] = rmse(a_ye[k:len(train)], train[k:len(train)])
                
            print("a_rmse[", fold, ",", k, "] = ", a_rmse[fold, k])
        
        a_rmse_mean[k] = np.mean(a_rmse[:, k], axis = 0)
        print("a_rmse_mean[", k, "] = ", a_rmse_mean[k])

    
    # Cáculo da média dos RMSE
    a_rmse_mean_validation = a_rmse_mean[:, 0]
    a_rmse_mean_train = a_rmse_mean[:, 1]
    
    # Cálculo do k-min
    y_train = a_rmse_mean_train[1:31]
    y_validation = a_rmse_mean_validation[1:31]
    x = a_k
    n_min = y_validation.argmin()
    # Retorna o index onde y é minimo que é o mesmo index de x. 
    # Com isso, achamos min_k
    min_k = x[n_min] 
    
    # Cálculo do valor estimado dos dados de teste
    n_train = k_fold * block
    n_total = np.size(a_y, 0)
    n_test = n_total - n_train
    a_y_test = a_y[(n_total - n_test): n_total]
    a_ye = np.zeros((n_test, 1))
    
    # Cálculo do w como todos os dados de teste
    a_w = init_array_w(k);
    a_phi = np.zeros((len(a_y_test), min_k + 1))
            
    # Cálculo do vetor phi baseado no dados de treinamento
    for i in range(k, len(a_y_test)):
        a_phi_x = np.array(phi(a_y_test, i, min_k))
        a_phi[i] = a_phi_x.T
    
    # Matriz phi.T . phi deve ser inversível
    assert(npl.matrix_rank(a_phi) == min(a_phi.shape[0], a_phi.shape[1]))
            
    # Solução fechada quando tem inversa
    a_w = npl.inv((a_phi.T).dot(a_phi)).dot(a_phi.T.dot(a_y_test))
            
    for i in range(min_k, n_test):
        ye = np.dot(phi(a_y_test, i, min_k).T, a_w)
        a_ye[i] = np.reshape(ye,1)
        
    
    #Plot dos graficos
    plt.subplot(2, 1, 1)
    plt.plot(x, y_train, '.--')
    plt.plot(x, y_validation, '.--')
    plt.title("IA006 - EFC1 - Computacional 1 - item a")
    plt.xlabel("hiperparâmetro - K")
    plt.ylabel("RMSE médio")
    plt.legend(['erro de treinamento', 'erro de validação'])
    plt.plot(x[n_min], y_validation[n_min], 'x')
    
    plt.subplot(2, 1, 2)

    y = a_ye[min_k:]
    x = np.array(range(n_train + min_k, n_train + n_test))
    plt.plot(x, y, '.--')
    
    y = a_y_test[min_k:]
    x = np.array(range(n_train + min_k, n_train + n_test))
    plt.plot(x, y, '.--')

    rmse_test = rmse(a_ye[min_k:], a_y_test[min_k:])
    
    plt.text(3300, 7, "RMSE = " + str(rmse_test))    
    plt.legend(['estimado', 'real'])
    plt.title("item b")
    plt.xlabel("dias desde " + str(start_time))
    plt.ylabel("temperatura - °C")
    
    plt.show()

    
