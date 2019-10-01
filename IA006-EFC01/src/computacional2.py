'''
Created on 21 de set de 2019

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


def get_w(num_attribute, a_y, lambda_value, a_p, delay):
    
    # A linha da matriz a_phi N x (T+1) é composta por [1 x_linha_1 ... x_linha_T]
    a_phi = np.zeros((len(a_y), num_attribute + 1))
    
    # Cálculo do vetor phi baseado no dados de treinamento
    #a_phi[delay:] = phi(a_y, i, a_p, delay)
    for i in range(delay, len(a_y)):
        a_phi_x = np.array(phi(a_y, i, a_p, delay))
        a_phi[i] = a_phi_x.T

    # Matriz phi.T . phi deve ser inversível
    # Nessa caso nao precisa, pois adiciona o termo de regularização
    # assert(npl.matrix_rank(a_phi) == min(a_phi.shape[0], a_phi.shape[1]))
    
    # Solução fechada quando tem inversa
    I_linha = np.identity(num_attribute + 1)
    I_linha[0,0] = 0
    # Matriz W  (T+1) x 1
    a_w = npl.inv((a_phi.T).dot(a_phi) 
                  + (lambda_value * I_linha)).dot(a_phi.T.dot(a_y))
    return a_w

# phi(a_y(a_n)) = [ 1 a_y(a_n-1) ... a_y(a_n-K) ]^T
def phi(a_y, n, w, delay):
    phi1 = np.ones((1,1))
    phi2 = w.dot(a_y[n-delay : n])
    np.tanh(phi2, phi2)    
    return np.concatenate([phi1, phi2])


if __name__ == '__main__':
    print("Iniciando main")
    
    # Numero de lambdas (Cross-validation com Ridge Regression) a serem gerados
    
    # [10^-5 10^-4 10^-3 10^-2 10^-1] 1 [10... 10^5]
    # nao pode ser muito pequene senao satura
    # 2*5 + 1  = 11 valores 
    # valores muito grande
    #a_lambda_g = np.geomspace(1e-2, 1e2, 5)
    #TESTAR COM LAMBA entre 1e6 1e-6
    a_lambda = np.geomspace(1e-6, 1e6, 13)
    
    #erro 0.000005 2.30
    #erro 0.1   - 2.33
    #erro 1     - 2.35
    #erro 10    - 2.30
    #erro 100   - 2.51
    #erro 1000  - 2.70
    #erro 10000 - 3.30
    
    #a_lambda = np.zeros(1)
    #a_lambda = np.concatenate([a_lambda, a_lambda_g])
    
    # Inicializa os arrays de dados com as temperaturas
    a_n, a_y, N, start_time = load_array_n_x()
    
    # Número de pastas a serem utilizadas 
    k_fold = 9
    kf = Kfolder(k_fold)
    
    # Matrix PHI
    a_phi = None
    
    # Range de valores para o T
    # ATENÇÃO GERAR ATÉ 101
    a_T = range(1, 101,1)
    
    # a ultima dimensão serve para guardar o erro de validacao e o erro de treinamento
    a_rmse = np.zeros((k_fold, len(a_T), a_lambda.shape[0],  2))
    
    # Matriz (T x lambda x 2) para guardar os da média RMSE 
    a_rmse_mean = np.zeros((len(a_T), a_lambda.shape[0], 2))
    
    # O bloco é calculado com o total dos dados de treino (treino + validacao) dividido pelo tamanho da pasta
    # ou o bloco é calculado com o total de dados menos a quantidade de dados do testes (365 dias)
    n_test = 365
    block = int((len(a_y[:,0]) - 365)/k_fold)
    
    # Número de atrasos, dado pela questão
    delay = 5
    
    # Parâmentro da distribuição uniforme.
    # Testar com o código abaixo para não explodir
    # delay = 5
    # numpy.tanh(numpy.dot(20*numpy.ones((1,delay)),numpy.random.rand(delay,1)/100))
    # a_p = np.random.rand(delay, 1)/pow(10,delay)
    random_b = 0
    # co
    #random_reduce_factor = 25
    random_reduce_factor = 50
    #a_p = np.zeros((max(a_T), delay))
    #for k in range(max(a_T)):
    #    a_p[k] = (np.random.rand(delay,)/random_reduce_factor) + random_b
        #a_p[k] = (np.random.uniform(-2, 2, delay)/random_reduce_factor) + random_b

    
    # Log de a_w dim=0 -> total de T, dim=1 -> total de lambdas, dim=2 -> total de w
    # Para um T, um Lambda, temos os paremtros  w com max(a_T) + 1 elemeto
    log_a_w = np.zeros((len(a_T), a_lambda.shape[0], max(a_T) + 1))
    
    # Para todo o t no range de 1 a 100
    for t in range(len(a_T)):
        for i_lambda in range(len(a_lambda)): 
            # Varre todas as pastas. 
            #COLOCAR A GERAÇÃO DO VETOR a_p AQUI
            for fold in range(k_fold):
                a_p = np.zeros((a_T[t], delay))
                for k in range(a_T[t]):
                    a_p[k] = (np.random.uniform(-1, 1, delay)/random_reduce_factor) + random_b
                
                print("Calculando para T = ", a_T[t], "fold = ", fold, "λ = ", a_lambda[i_lambda])

                # Recupera dados de treinamento e validação
                train, test = kf.split_data(a_y[0: k_fold*block,0])
                
                a_ye = np.zeros((len(train), 1))
                
                # Calcula w para o critério de minimizar o erro quadrático minimo.
                a_w = get_w(a_T[t], train, a_lambda[i_lambda], a_p, delay)
                 
                # Cálculo do erro em relação ao dados de validação (test)
                for i in range(delay, len(test)):
                    ye = np.dot(phi(test, i, a_p, delay).T, a_w)
                    a_ye[i] = np.reshape(ye,1)
                
                # Cálculo da RMSE para um t específico.
                a_rmse[fold, t, i_lambda, 0] = rmse(a_ye[delay:len(test)], test[delay:len(test)])
                
                # Cálculo do erro em relação ao dados de treinamento (train)
                for i in range(delay, len(train)):
                    ye = np.dot(phi(train, i, a_p, delay).T, a_w)
                    a_ye[i] = np.reshape(ye,1)
                    
                # Cálculo da RMSE 
                a_rmse[fold, t, i_lambda, 1] = rmse(a_ye[delay:len(train)], train[delay:len(train)])
                
                # Registro do w
                for i in range(len(a_w)):
                    log_a_w[t, i_lambda, i] = a_w[i]
                
                print("a_rmse[", fold, ",", t, "] = ", a_rmse[fold, t])
            
            # Cálculo da média dos RMSE dos foldeer
            # Matriz (T x lambda x 2) para guardar os da média RMSE 
            a_rmse_mean[t, i_lambda ] = np.mean(a_rmse, axis=0)[t, i_lambda]
            
        #print("Log a_w = (iT,iλ) = " , t, i_lambda, log_a_w[t, i_lambda])
 
    #Plot dos graficos
    fig1, axs = plt.subplots(2, 1, constrained_layout=False)
    ax1 = axs[0]
    ax2 = axs[1]
    
    # Pega os valores de rmse de validacao. O resultado sera um vetor T x Lambda
    # Daí, para cada T, achar o minimo de lambda (axis = 1)
    # Matriz (T x lambda x 2) para guardar os da média RMSE 
    y_validation = a_rmse_mean[:, :, 0].min(axis = 1) 
    y_train = a_rmse_mean[:, :, 1].min(axis = 1) 
    # x = [ 1 2 3 4 ... 100]
    x = a_T

    i_t_min = y_validation.argmin()
    T_min = x[i_t_min]
    rmse_min = y_validation[i_t_min]
    
    ax1.plot(x, y_train, '.--')
    ax1.plot(x, y_validation, '.--')
    
    
    note_ax1 = ("Distribuição uniforme - [-0,02, 0,02)\n"
             + "RMSE mímimo = " + str(rmse_min)  + "\n"
             + "Mínimo para T = " + str(T_min)  + "\n")
    print(note_ax1)
    
    ax1.text(50, 2.6, note_ax1)
    #ax1.annotate(note_ax1, xy=(T_min, rmse_min), xytext=(T_min + 0.5, rmse_min + ),
    #        arrowprops=dict(facecolor='black', shrink=0.05),)
    
    ax1.plot(x[i_t_min], y_validation[i_t_min], 'x')
    
    ax1.legend(['erro de treinamento', 'erro de validação'])
    ax1.set_title("IA006 - EFC1 - Computacional 2 - item a ")
    ax1.set_xlabel("número de atributos T")
    ax1.set_ylabel("RMSE médio")
     
     
    # Pega os valores de rmse de validacao. O resultado sera um vetor T x Lambda com entrada
    # representa o RMSE
    # Matriz (T x lambda x 2) para guardar os da média RMSE 
    rmse_t_lambda = a_rmse_mean[:, :, 1]

    y_lambda = np.zeros((len(a_T),))
    for t in range(len(a_T)):
        
        # Pega o index da linha que tem o menor RMSE
        i_lambda_min_rmse = np.where( rmse_t_lambda[t] == np.amin(rmse_t_lambda[t]))
        y_lambda[t] = a_lambda[i_lambda_min_rmse][0]
    
    ax2.plot(x, y_lambda, '.--')
    print("λs minimos para cada T = ", y_lambda)
    ax2.set_title("Item b")
    ax2.set_xlabel("número de atributos - T")
    ax2.set_ylabel("fator de regularização - λ")
    ax2.set_yscale("log")
    
    # Criando
    fig2, axs = plt.subplots(2, 1, constrained_layout=False)
    ax4 = axs[0]
    ax3 = axs[1]
    
    # Retorna a tupla com o index do lmabda que tem o menor RMSE. [0] linha [1] coluna
    # Pode ser que o terno seja mais de um valor
    # Matriz (T x lambda x 2) para guardar os da média RMSE 
    i_lambda_min_rmse = np.where( rmse_t_lambda == np.amin(rmse_t_lambda))[1]
    
    
    print("array p = ", a_p[i_t_min])
    print("t index, ", i_t_min, ", T min = ", T_min, ", λ minimo = ", a_lambda[i_lambda_min_rmse], ", rmse_min = ", rmse_min)
    
    
    # Cálculo do valor estimado dos dados de teste
    n_train = k_fold * block
    n_total = np.size(a_y, 0)
     
    a_y_test = a_y[(n_total - n_test): n_total]
    a_ye = np.zeros((n_test, 1))
    
    # Calculano o vetor de paramentros aleatorios
    a_p = np.zeros((T_min, delay))
    for k in range(T_min):
        a_p[k] = (np.random.uniform(-1, 1, delay)/random_reduce_factor) + random_b

    # Calculando novo w baseado no lambda minimo e T minimo
    a_w = get_w(T_min, a_y[:(n_total - n_test)], a_lambda[i_lambda_min_rmse[0]], a_p, delay)
    for i in range(delay, n_test):
        ye = np.dot(phi(a_y_test, i, a_p, delay).T, a_w)
        a_ye[i] = np.reshape(ye,1)
    
    # Plot da curva estimada
    y_estimado = a_ye[delay:] 
    x = np.array(range(n_train + delay, n_train + n_test))
    ax3.plot(x, y_estimado, '.--')

    # Plot da curva real
    y_real = a_y_test[delay:]
    x = np.array(range(n_train + delay, n_train + n_test))
    ax3.plot(x, y_real, '.--')
    
    # Calculo do RMSE do teste para o melhor lambda
    rmse_test = rmse(y_estimado, y_real)
    ax3.text(3300, 7, "RMSE = " + str(rmse_test) + "\n"
             + "λ = " + str(a_lambda[i_lambda_min_rmse[0]]) + "\n")
    ax3.legend(['estimado', 'real'])
    ax3.set_title("Item c")
    ax3.set_xlabel("dias desde " + str(start_time))
    ax3.set_ylabel("temperatura - °C")
    
    # Aplique o modelo com os melhores valores de (regularização) e de aos dados de teste.
    # Meça o desempenho em termos de RMSE.
    
    a_rmse_best_lambda = np.zeros((len(a_T),))
    for t in range(len(a_T)):
        # Calculano o vetor de paramentros aleatorios
        a_p = np.zeros((a_T[t], delay))
        for k in range(a_T[t]):
            a_p[k] = (np.random.uniform(-1, 1, delay)/random_reduce_factor) + random_b
    
        # Calculando novo w baseado no lambda minimo e T minimo
        a_w = get_w(a_T[t], a_y[:(n_total - n_test)], y_lambda[t], a_p, delay)
        for i in range(delay, n_test):
            ye = np.dot(phi(a_y_test, i,  a_p, delay).T, a_w)
            a_ye[i] = np.reshape(ye,1)
        
        a_rmse_best_lambda[t] = rmse(a_ye[delay:], a_y_test[delay:])
    
    ax4.plot(a_T, a_rmse_best_lambda, '.--')
    ax4.set_title("Item c")
    ax4.set_xlabel("número de atributos T")
    ax4.set_ylabel("RMSE médio com dados teste")
    
#     ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
#     ax5.plot(a_T, y_lambda, '.--', color = 'tab:orange')
#     print("λs minimos para cada T = ", y_lambda)
#     ax5.set_xlabel("número de atributos - T")
#     ax5.set_ylabel("fator de regularização - λ")
#     ax5.set_yscale("log")

    plt.show()

    
