'''
Created on 21 de set de 2019

@author: navarro
'''
import csv
import numpy as np
import numpy.linalg as npl
from datetime import datetime, date
import matplotlib.pyplot as plt
from golden_section_search import gr


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def f_rmse(lambda_value, train, test):
    print("Calculando para T = ", a_T[t], "fold = ", fold, "lambda = ", lambda_value)

    a_w = get_w(a_T[t], train, lambda_value, a_p, delay)
     
    # Cálculo do erro em relação ao dados de validação (test)
    for i in range(delay, len(test)):
        ye = np.dot(phi(test, i, a_T[t], a_p, delay).T, a_w)
        a_ye[i] = np.reshape(ye,1)
    
    # Cálculo da RMSE para um t específico.
    return rmse(a_ye[delay:len(test)], test[delay:len(test)]), a_w
    
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
    for i in range(delay, len(a_y)):
        a_phi_x = np.array(phi(a_y, i, num_attribute, a_p, delay))
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
def phi(a_y, n, T, w, delay):
    
    phi1 = np.ones((1,1))
    phi2 = np.ones((T,1))
    
    x_r = a_y[n-delay:n]
    for k in range(T):
        x_linha_k = np.tanh(x_r.T.dot(w[k]))
        phi2[k,0] = x_linha_k 
    
    return np.concatenate([phi1, phi2])


if __name__ == '__main__':
    print("Iniciando main")
    
    # Numero de lambdas (Cross-validation com Ridge Regression) a serem gerados
    #num_regularisation_lambda = 2
    
    #a_lambda_g = np.geomspace(1/pow(10,num_regularisation_lambda - 1 ), 10, num_regularisation_lambda)
    #a_lambda_g = np.geomspace(0.2, pow(10,num_regularisation_lambda - 1 ), num_regularisation_lambda)
    #a_lambda_g = np.arange(0, 1, 0.5)
    a_lambda = np.zeros(1)
    #a_lambda = np.concatenate([a_lambda, a_lambda_g])
    lambda_lower = 0
    lambda_upper = 1
    gss_tol=1e-4
    
    # Inicializa os arrays de dados com as temperaturas
    a_n, a_y, N, start_time = load_array_n_x()
    
    # Número de pastas a serem utilizadas 
    k_fold = 2
    kf = Kfolder(k_fold)
    
    # Matrix PHI
    a_phi = None
    
    # Range de valores para o T
    a_T = range(1,100, 1)
    #a_T = np.geomspace(1, 100, 9)
    
    # a ultima dimensão serve para guardar o erro de validacao e o erro de treinamento
    # e a penultima para guarda o valor de lambda
    a_rmse = np.zeros((k_fold, len(a_T), 3))
    
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
    #random_reduce_factor = 25
    random_reduce_factor = 50
    a_p = np.zeros((max(a_T), delay))
    for k in range(max(a_T)):
        a_p[k] = (np.random.rand(delay,)/random_reduce_factor) + random_b

    
    # Log de a_w dim=0 -> total de T, dim=1 -> total de lambdas, dim=2 -> total de w
    # Para um T, um Lambda, temos os paremtros  w com max(a_T) + 1 elemeto
    log_a_w = np.zeros((len(a_T), max(a_T) + 1))
    
    # Varre todas as pastas. 
    for fold in range(k_fold):
        train, test = kf.split_data(a_y[0: k_fold*block,0])
        a_ye = np.zeros((len(train), 1))
        
        # Para todo o t no range de 1 a 100
        for t in range(len(a_T)):
            
            # Iniciar golder section search
            a = lambda_lower
            b = lambda_upper
            c = b - (b - a)/gr
            d = a + (b - a)/gr
            
            while(np.abs(c - d) > gss_tol):
                rmse_c, a_w_c = f_rmse(c, train, test)
                rmse_d, a_w_d = f_rmse(d, train, test)
                
                if(rmse_c < rmse_d):
                    b = d
                else:
                    a = c
                    
                c = b - (b - a)/gr
                d = a + (b - a)/gr
            
            # Achar valor do a_w  e rmse baseado no lambda que leva ao erro minimo de validação
            gss_lambda = (a + b)/2 
            rmse_validation_with_lambda_min, a_w = f_rmse(gss_lambda, train, test)
            
            # Cálculo do erro em relação ao dados de treinamento (train)
            for i in range(delay, len(train)):
                ye = np.dot(phi(train, i, a_T[t], a_p, delay).T, a_w)
                a_ye[i] = np.reshape(ye,1)
            

            a_rmse[fold, t] = (gss_lambda, rmse_validation_with_lambda_min, rmse(a_ye[delay:len(train)], train[delay:len(train)]))
            
            # Registro do w
            for i in range(len(a_w)):
                log_a_w[t, i] = a_w[i]
            
            print("a_rmse[", fold, ",", t, "] = ", a_rmse[fold, t])
            

    # Cáculo da média dos RMSE
    # TODO VERIFICAR NO CODIGO ONDE ESTA USANDO O  
    # pode ser que 
    a_rmse_mean = np.mean(a_rmse, axis=0)
 
    #Plot dos graficos
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    
    y_train = a_rmse_mean[:, 2] 
    y_validation = a_rmse_mean[:, 1] 

    # x = [ 1 2 3 4 ... 100]
    x = a_T

    i_t_min = y_validation.argmin()
    T_min = x[i_t_min]
    rmse_min = y_validation[i_t_min]
    
    ax1.plot(x, y_train, '.-')
    ax1.plot(x, y_validation, '.-')
    
    ax1.text(50, 2.6, "Distribuição uniforme - (0," + str(1/random_reduce_factor) + "]\n"  
             + "RMSE mímimo =" + str(rmse_min)  + "\n"
             + "T mímimo = " + str(T_min)  + "\n")  
    
    ax1.plot(x[i_t_min], y_validation[i_t_min], 'x')
    
    ax1.legend(['erro de treinamento', 'erro de validação'])
    ax1.set_title("IA006 - EFC1 - Computacional 2 - item a ")
    ax1.set_xlabel("número de atributos T")
    ax1.set_ylabel("RMSE médio")
     

     
    # Cálculo do valor estimado dos dados de teste
    n_train = k_fold * block
    n_total = np.size(a_y, 0)
     
    a_y_test = a_y[(n_total - n_test): n_total]
    a_ye = np.zeros((n_test, 1))
    
    # Pega os valores de rmse de validacao. O resultado sera um vetor T x Lambda com entrada
    # representa o RMSE
    
    ## PEGAR 
    y_rmse_with_min_lambda = a_rmse_mean[ :, 0]

    
    ax2.plot(x, y_rmse_with_min_lambda, '.-')
    print("λs minimos = ", y_rmse_with_min_lambda)
    
    ax2.set_title("Item b")
    ax2.set_xlabel("número de atributos - T")
    ax2.set_ylabel(r'fator de regularização - λ')
    ax2.set_yscale("log")
    lambda_min_rmse = y_rmse_with_min_lambda.min()
    
    print("array p = ", a_p[i_t_min])
    print("t index, ", i_t_min, "T min = ", T_min, "λ minimo = ", lambda_min_rmse, "rmse_min = ", rmse_min)
    
    # Calculando novo w baseado no lambda minimo e T minimo
    a_w = get_w(T_min, a_y[:(n_total - n_test)], lambda_min_rmse, a_p, delay)
    for i in range(delay, n_test):
        ye = np.dot(phi(a_y_test, i, T_min, a_p, delay).T, a_w)
        #ye = np.dot(phi(a_y_test, i, T_min, a_p, delay).T, min_a_w)
        a_ye[i] = np.reshape(ye,1)
         
    y_estimado = a_ye[delay:] 
    x = np.array(range(n_train + delay, n_train + n_test))

    
    ax3.plot(x, y_estimado, '.-')
      
    y_real = a_y_test[delay:]
    x = np.array(range(n_train + delay, n_train + n_test))
    ax3.plot(x, y_real, '.-')
    
    rmse_test = rmse(y_estimado, y_real)
    
    
    ax3.text(3300, 7, "RMSE = " + str(rmse_test) + "\n"
             + "λ = " + str(lambda_min_rmse) + "\n")
             
    ax3.legend(['estimado', 'real'])
    ax3.set_title("Item c")
    ax3.set_xlabel("dias desde " + str(start_time))
    ax3.set_ylabel("temperatura - °C")
    
    plt.show()

    
