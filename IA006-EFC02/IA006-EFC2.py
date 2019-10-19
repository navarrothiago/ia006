# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parte 1 - Classificação binária

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from IPython.core.debugger import set_trace
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %%
import pandas as pd
import matplotlib.pyplot as plt
import pdb;
import numpy as np

# %%
from src.efc2 import csv_reader
# #%pycat efc2/csv_reader.py

# %%
data = csv_reader.load_csv()
#data.head(data.shape[0])

data = data.drop("Unnamed: 0", 1)
data.head(data.shape[0])

# %% [markdown]
# ## Item a

# %%
columns = list(data)

plot_rows = int(len(columns) / 2)
if(len(columns) % 2 != 0):
    plot_rows = plot_rows + 1

print(plot_rows)
fig1, axs = plt.subplots(plot_rows, 2, constrained_layout=True, figsize=(20,30))

colors = ['red', 'green']
labels = data.label.unique
#data.hist(ax=axs, label=labels, bins=100);
_= data.loc[data['label'] == 1].hist(ax=axs, bins=100, label='1', color="royalblue", density = False, alpha = 0.7);
_= data.loc[data['label'] == 0].hist(ax=axs, bins=100,  label='0', color="orange", density = False,  alpha = 0.7);

# %% [markdown]
# ##### ?data.corr

# %% {"jupyter": {"outputs_hidden": true}}
f = plt.figure(figsize=(30, 30))
plt.matshow(data.corr(), fignum=f.number)

for (i, j), z in np.ndenumerate(data.corr()):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color="r")
    
plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
plt.margins(x=0, y=0);

# %% [markdown]
# ## Item b

# %%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np

train, test = train_test_split(data, test_size=0.2)


# %%
def gradient_descent(matrix_y, matrix_ye, matrix_phi):
    matrix_error = matrix_y - matrix_ye
    return -matrix_error.T.dot(matrix_phi)/len(matrix_y)


# %%
# x é o phi_x.T
def phi(x):
    phi1 = np.ones((x.shape[0],1))
    return np.concatenate((phi1, x), axis=1)


# %%
# Função custo
def j_cross_entropy(targets, predictions, epsilon=1e-12):
        
    # só funciona se epsilon for maior que zero
    assert(epsilon > 0)
    
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    
    #pdb.set_trace()
    ce_when_y_1 = -np.sum(targets*np.log(predictions))/N
    ce_when_y_0 = - np.sum((1-targets)*np.log(1-predictions))/N
    ce = ce_when_y_1 + ce_when_y_0
    
    #print("y = 1 ", ce_when_y_1, " y = 0", ce_when_y_0, " total = ", ce)
    
    return ce


# %% [markdown]
# ### Fase de Treinamento

# %%
#verificar se pode estar saturando sigmoide.
matrix_phi = phi(preprocessing.scale(train.drop("label", 1).values))

# matrix com os dados 
matrix_y = train["label"].values.reshape(train["label"].values.shape[0], 1)

matrix_phi
#matrix_y

# %%
from sklearn.metrics import log_loss
# Acha o w pelo método grandiente descendente e retorna, também, o custo em cada iteração
def find_w(y_train, phi, alpha, iterations):
    
    # A dimensão da matrix w é número de atributos mais 1 x 1
    w = np.random.rand(phi.shape[1], 1)

    matrix_cost = np.zeros((iterations,))

    for i in range(iterations):
        z = phi.dot(w)
        matrix_ye = 1 /(1 + np.exp(-z))

        w = w - (alpha) * gradient_descent(y_train, matrix_ye, phi).T
        #matrix_cost[i] = j_cross_entropy(y_train, matrix_ye)
        matrix_cost[i] = log_loss(y_train, matrix_ye)

    df_cost = pd.DataFrame(matrix_cost,columns=['cost'])
    return w, df_cost


# %%
matrix_w, df_cost = find_w(matrix_y, matrix_phi, 0.01, 10000)
df_cost.plot();

# %% [markdown]
# ### Fase de Teste
#

# %%
#verificar se pode estar saturando sigmoide.
matrix_phi_test = phi(preprocessing.scale(test.drop("label", 1).values))

# matrix com os dados de teste
# recupera dados com rotulos label e reshape 
matrix_y_test = test["label"].values.reshape(test["label"].values.shape[0], 1)

# calculando estimativa para todos os dados de teste com o w calculado anteriormente
z = matrix_phi_test.dot(matrix_w)
matrix_ye_test = 1 /(1 + np.exp(-z))

# %% [markdown]
# ### Fase de decisão 

# %%
from sklearn.metrics import confusion_matrix

thresholds = np.arange(0, 1.01, 0.01)
confusion_matrix_array = np.zeros((len(thresholds), 2 ,2))

# definindo threshold
for i in range(len(thresholds)):
    
    # Decisão: coloca (decide por) 1 se for maior, c.c. 0
    matrix_ye_test_decided = (matrix_ye_test >= thresholds[i]).astype(int)

    confusion_matrix_array[i] = confusion_matrix(matrix_y_test, matrix_ye_test_decided)

confusion_matrix_array

# %% [markdown]
# ### ROC

# %% [markdown]
# **from sklearn import metrics**

# %%
from sklearn import metrics

fpr, tpr, n_thresholds = metrics.roc_curve(matrix_y_test, matrix_ye_test)

_= plt.plot(fpr, tpr, '.--', label="Classificador");
_= plt.title("Receiver operating curve - ROC");
_= plt.xlabel("false positive - %");
_= plt.ylabel("recall - true positive - %");
_= plt.legend();


# %% [markdown]
# **na mão**

# %%
# Curva ROC
# x - falso positivo (fp / tn + fp ) = (fp / N-) 
# y - recall - sensibilidade (tp / tp + fn) - true positive
pe_ = confusion_matrix_array[:, 0,1]/(confusion_matrix_array[:, 0,0] + confusion_matrix_array[:, 0,1])
recall = confusion_matrix_array[:, 1,1]/(confusion_matrix_array[:, 1,1] + confusion_matrix_array[:, 1,0])

plt.plot(pe_, recall, '.--');
plt.title("Receiver operating curve - ROC");
plt.xlabel("false positive - %");
plt.ylabel("recall - true positive - %");


# %%
# %connect_info

# %%
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(recall, bins=20, color='#0504aa',
                            alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Recall')
plt.ylabel('Frequency')
plt.title('Recall Histogram')
#plt.text(0.2, 60, r'$\mu=15, b=3$')
maxfreq = n.max()
n
bins
maxfreq/len(recall)
len(recall)
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# %%
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(pe_, bins=10, color='#0504aa',
                            alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('False Positive')
plt.ylabel('Frequency')
plt.title('False Positive Histogram')
#plt.text(0.2, 60, r'$\mu=15, b=3$')
maxfreq = n.max()
n
bins
maxfreq/len(pe_)
len(pe_)
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# %% [markdown]
# ### F-score 
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/400px-Precisionrecall.svg.png"
#      alt="F-score"
#      style="float: left; margin-right: 400px;" />

# %%
# Proporção de padrões da classe positiva corretamente classificados em 
# relação a todos os exemplos atribuídos à classe positiva.
precision = confusion_matrix_array[:, 1,1]/np.clip(confusion_matrix_array[:, 1,1] + confusion_matrix_array[:, 0,1], 1e-12, None)

# Do total de verdadeiro positivo - Proporção de amostras da classe positiva corretamente classificadas. 
recall = confusion_matrix_array[:, 1,1]/np.clip(confusion_matrix_array[:, 1,1] + confusion_matrix_array[:, 1,0], 1e-12, None)

pd.DataFrame({"precisao": precision, "recall":recall})

# %%
# Workaround para nao dá divisão por zero. O ideal era tratar esses casos separadamente.
recall = np.clip(recall, 1e-12, None)
precision = np.clip(precision, 1e-12, None)

# %%
m = 1
f_score = ((m + 1)*recall*precision)/(recall + m*precision)

index_max = np.argmax(np.array(f_score[:-1]))
# Workaround zero divido por zero. O ideal era tratar esse caso antes.
ax = plt.plot(thresholds[:-1], f_score[:-1], '.--');
plt.plot(thresholds[index_max], f_score[index_max], 'X');
plt.annotate("Maximum in (" + str(np.around(thresholds[index_max], decimals=2)) 
            + ", " + str(np.around(f_score[:-1][index_max], decimals=3)) + ")",
            #xy = (matrix_confusion_df['threshold'].iloc[index]/2, f_score.iloc[index]/2))
             xy = (0.2, 0.8))
plt.title("F-score evolution");
plt.xlabel("threshold - un");
plt.ylabel("F-score - un");

# Para m = 1
# Valores  de bem próximos de 1 indicam que o classificador  obteve  
# bons resultados tanto na precisão quanto no recall.
# %%
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(f_score[:-1], bins=10, color='#0504aa',
                            alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('F1-score')
plt.ylabel('Frequency')
plt.title('F1-score Histogram')
#plt.text(0.2, 60, r'$\mu=15, b=3$')
maxfreq = n.max()
n
bins
maxfreq/len(f_score[:-1])
len(f_score[:-1])
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# %% [markdown]
# ## Item c 
#
# Obter matrix de confusão

# %%
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(constrained_layout=False)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    ## Bug ??? - sem isso não plota corretamente
    ax.margins(x=0, y=0)
    ##############
    
    return ax


# %%
 from sklearn.utils.multiclass import unique_labels
    
plot_confusion_matrix(matrix_y_test.astype(int), (matrix_ye_test >= thresholds[index_max]).astype(int), unique_labels([0,1]));

# %% [markdown]
# # Parte 2 - Classificação multi-classe
#
# ## Item a
#
# Técnica adotada: **Um contra todos**
#
# A ideia aqui é fazer 5 classificadores, um para cada classe.

# %%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Ler Datasets e converte para float cada entrada

# %%
#text_file = open("har_smartphone/X_train.txt", "r")
# with open("har_smartphone/X_train.txt", "r") as my_file:
#   for line in my_file:
#       row = [float(x) for x in line.split()]
#       print(str)

X_train = pd.read_fwf('har_smartphone/X_train.txt', header=None)
X_train

# %%
X_train = pd.read_fwf('har_smartphone/X_train.txt', header=None)
y_train = pd.read_fwf('har_smartphone/y_train.txt', header=None)
X_test = pd.read_fwf('har_smartphone/X_test.txt', header=None)
y_test = pd.read_fwf('har_smartphone/y_test.txt', header=None)

# %% [markdown]
# Após separar os dataset de treinamento e teste, iremos achar os w dos 5 classificadores usando a estrutura de regressão logística que minimize o critério da função de custo, _cross-entropy_ .

# %%
# matrix com os atributos
matrix_phi = phi(preprocessing.scale(X_train))

# matrix com os dados de validação
matrix_y = y_train.values

# Número de classificadores e seus parâmetros
Q = 6
matrix_w = np.zeros((Q,X_train.shape[1] + 1))

# %% [markdown]
# ### Fase de Treinamento
# Para achar os ws, os dados de y devem ser transformados. Como será implementada uma classificação um contra todos, para cada iteração q, sendo q igual ao label do Classificador k, k = 0, ... Q-1:
# * Se q == k, então o label que identifica a classe k passa a ser 1
# * Se q != k, então o label que identifica a classe k passa a ser 0

# %%
fig1, axs = plt.subplots(2, 3, constrained_layout=True)

for q in range (Q):
    # Se for classe q, label -> 1, cc label -> 0
    matrix_y_tranform = (matrix_y == q + 1).astype(int)
    w, df_cost = find_w(matrix_y_tranform, matrix_phi, 0.01, 10000)
    matrix_w[q,:] = np.array(w)[:,0]
    _ = axs[q//3, q%3].set_title("Classificador para classe: " + str(q))
    _ = df_cost.plot(ax=axs[q//3, q%3], figsize=(10, 10));

# %% [markdown]
# ### Fase de Teste
# Calcular a estimativa em cada classificador com os dados de teste
#
# Matriz w

# %%
# matrix com os atributos
matrix_phi_test = phi(preprocessing.scale(X_test))

# matrix com os dados de teste mais 1, para nao deixar nenhum nulo
matrix_y_test = y_test.values

# calculando estimativa para todos os dados de teste com o w calculado anteriormente
matrix_ye_test = np.zeros((matrix_y_test.shape[0], Q))
for q in range(Q):
    z = matrix_phi_test.dot(matrix_w[q])
    matrix_ye_test[:, q] = 1 /(1 + np.exp(-z))

# %%
pd.DataFrame(matrix_ye_test)
pd.DataFrame(matrix_y_test)

# %% [markdown]
# ### Fase de Decisão
#
# Calculando matriz de confusão para cada threshold 
# **sem toolbox**

# %%
matrix_ye_test_decided = np.zeros((matrix_y_test.shape[0], Q))

#thresholds = np.arange(0, 1.01, 0.1)

index = pd.MultiIndex.from_product([range(Q), thresholds], names=['classificador', 'threshold'])
matrix_confusion_df = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'], index=index)

for q in range(Q):
    # definindo threshold
    for threshold in thresholds:

        # Decisão: coloca (decide por) 1 se for maior, c.c. 0
        matrix_ye_test_decided[:, q] = (matrix_ye_test[:, q] >= threshold).astype(int)

        matrix_confusion = [{'tp':0, 'tn':0, 'fp':0, 'fn':0}]
        row_df = pd.DataFrame(matrix_confusion)

        #matrix_y_test.T, matrix_ye_test.T

        for y, ye in zip((matrix_y_test == q + 1).astype(int), matrix_ye_test_decided[:, q]):
            if(y == ye):
                if(y == 1):
                    row_df["tp"] =  row_df["tp"] + 1
                else:
                    row_df["tn"] =  row_df["tn"] + 1
            else:
                if(y == 1):
                    # ye == 0, porem y == 0
                    row_df["fn"] =  row_df["fn"] + 1
                else:
                    # ye == 1, porem y == 0
                    row_df["fp"] =  row_df["fp"] + 1

        matrix_confusion_df.loc[q, threshold] = row_df.iloc[0]

matrix_confusion_df

# %% [markdown]
# Calculando matriz de confusão para cada threshold **com toolbox**

# %%
matrix_ye_test_decided = np.zeros((matrix_y_test.shape[0], Q))
cm_one_against_rest_array = np.zeros((Q, len(thresholds), 2, 2))

for q in range(Q):
    for i in range(len(thresholds)):
        
        # Decisão: coloca (decide por) 1 se for maior, c.c. 0
        matrix_ye_test_decided[:, q] = (matrix_ye_test[:, q] >= thresholds[i]).astype(int)
        
        # Matriz de confusão
        cm_one_against_rest_array[q,i] = confusion_matrix((matrix_y_test == q + 1).astype(int), matrix_ye_test_decided[:, q])

# %% [markdown]
# ### ROC
# Com toolbox

# %%
from sklearn import metrics
for q in range (Q):
    # Decisão: coloca (decide por) 1 se for maior, c.c. 0
    fpr, tpr, n_thresholds = metrics.roc_curve((matrix_y_test == q + 1).astype(int), matrix_ye_test[:, q])
    
    _ = plt.plot(fpr, tpr, '.--', label="Classificador " + str(q));
    _ = plt.title("Receiver operating curve - ROC");
    _ = plt.xlabel("false positive - %");
    _ = plt.ylabel("recall - true positive - %");
    _ = plt.legend();

# %%
q = 0
cm_one_against_rest_array[q, :,0,1]/(cm_one_against_rest_array[q, :,0,0] + cm_one_against_rest_array[q, :,0,1])
cm_one_against_rest_array[q, :,1,1]/(cm_one_against_rest_array[q, :,1,1] + cm_one_against_rest_array[q, :,1,0])

# %% [markdown]
# Sem toolbox

# %%
# Curva ROC
# x - falso positivo (fp / tn + fp ) = (fp / N-) 
# y - recall - sensibilidade (tp / tp + fn) - true positive
#from IPython.core.debugger import set_trace
# #%debug
for q in range (Q):
    pe_ = cm_one_against_rest_array[q, :,0,1]/(cm_one_against_rest_array[q, :,0,0] + cm_one_against_rest_array[q, :,0,1])
    recall = cm_one_against_rest_array[q, :,1,1]/(cm_one_against_rest_array[q, :,1,1] + cm_one_against_rest_array[q, :,1,0])
    #set_trace()
    _ = plt.plot(pe_, recall, '.--', label="Classificador " + str(q));
    _ = plt.title("Receiver operating curve - ROC");
    _ = plt.xlabel("false positive - %");
    _ = plt.ylabel("recall - true positive - %");
    _ = plt.legend();

_ = plt.show()

# %%
# Curva ROC
# x - falso positivo (fp / tn + fp ) = (fp / N-) 
# y - recall - sensibilidade (tp / tp + fn) - true positive
for q in range (Q):
    pe_ = matrix_confusion_df.loc[q]['fp']/(matrix_confusion_df.loc[q]['tn'] + matrix_confusion_df.loc[q]['fp'])
    recall = matrix_confusion_df.loc[q]['tp']/(matrix_confusion_df.loc[q]['tp'] + matrix_confusion_df.loc[q]['fn'])

    _ = plt.plot(pe_, recall, '.--', label="Classificador " + str(q));
    _ = plt.title("Receiver operating curve - ROC");
    _ = plt.xlabel("false positive - %");
    _ = plt.ylabel("recall - true positive - %");
    _ = plt.legend();

plt.show()

# %% [markdown]
# ### F-score

# %%
index_max_array = np.zeros((Q))
index_max
for q in range(Q):
    # Proporção de padrões da classe positiva corretamente classificados em 
    # relação a todos os exemplos atribuídos à classe positiva.
    precision = cm_one_against_rest_array[q, :, 1,1]/np.clip(cm_one_against_rest_array[q, :, 1,1] + cm_one_against_rest_array[q, :, 0,1], 1e-12, None)

    # Do total de verdadeiro positivo - Proporção de amostras da classe positiva corretamente classificadas. 
    recall = cm_one_against_rest_array[q, :, 1,1]/np.clip(cm_one_against_rest_array[q, :, 1,1] + cm_one_against_rest_array[q, :, 1,0], 1e-12, None)

    # Workaround para nao dá divisão por zero. O ideal era tratar esses casos separadamente.
    recall = np.clip(recall, 1e-12, None)
    precision = np.clip(precision, 1e-12, None)

    m = 1
    f_score = ((m + 1)*recall*precision)/(recall + m*precision)

    # Pega o index do F-score máximo.
    index_max = np.argmax(np.array(f_score[:-1]))
    
    # Plota removendo os NaN
    axs = plt.plot(thresholds[:-1], f_score[:-1], '.--', label="Classificador " + str(q));
    _= plt.plot(thresholds[:-1][index_max], f_score[:-1][index_max], 'X', color=axs[0].get_color());
    _= plt.title("F-score evolution");
    _= plt.xlabel("threshold - un");
    _= plt.ylabel("F-score - un");
    _= plt.legend()
    
    index_max_array[q] = index_max

# %% [markdown]
# Calculando a matriz de confusão para as 6 classes adorando o seguinte critério de decisão:
#
# $g(\textbf{x}) = arg\,max_{i}(g_{i}(\textbf{x}))$
#
# onde $g_{i}$ é a função discriminante do i-ésimo classificador. Ver referência [MIT - multiclass.pdf](https://www.mit.edu/~9.520/spring09/Classes/multiclass.pdf).

# %%
from sklearn.metrics import confusion_matrix

classes = np.array(range(Q))
np.set_printoptions(precision=2)
plot_confusion_matrix(matrix_y_test - 1, np.argmax(matrix_ye_test, axis=1), classes)


# %% [markdown]
# Calcular o a métrica global para a avaliação do desempenho (médio) deste classificador.
#
# [Referência - A systematic analysis of performance measures for classification tasks](http://atour.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf)
#
# [Referência - Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
#
# The confusion elements for each class are given by:
#
# $tp_i = c_{ii}$
#
# $fp_i = \sum_{l=1}^L c_{li} - tp_i$
#
# $fn_i = \sum_{l=1}^L c_{il} - tp_i$
#
# $tn_i = \sum_{l=1}^L \sum_{k=1}^L c_{lk} - tp_i - fp_i - fn_i$
#
#
# <img src="https://devopedia.org/images/article/208/6541.1566280388.jpg"
#      alt="Matriz de confusão multi-class"
#      style="float: left; margin-right: 400px; width:500px;height:300px" />
#

# %%
from sklearn.metrics import f1_score

f1_score(matrix_y_test - 1,  np.argmax(matrix_ye_test, axis=1), average='macro')
f1_score(matrix_y_test - 1,  np.argmax(matrix_ye_test, axis=1), average='micro')

# %% [markdown]
# ### Item b
#
# kNN

# %%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import csv
import pandas as pd

X_train = pd.read_fwf('har_smartphone/X_train.txt', header=None)
y_train = pd.read_fwf('har_smartphone/y_train.txt', header=None)
X_test = pd.read_fwf('har_smartphone/X_test.txt', header=None)
y_test = pd.read_fwf('har_smartphone/y_test.txt', header=None)


X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# %% [markdown]
# Definindo a classe KNNClassifier

# %%
import numpy as np
from IPython.core.debugger import set_trace

class KNNClassifier:
    
    __X_test = np.array(None)
    __dist = np.array(None)
    
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        
    def __vote(self, y_predict, classes):
        votes = np.zeros((len(classes)))
        
        for i in range(len(y_predict)):
            for j in range(len(classes)):
                if(classes[j] == y_predict[i]):
                    votes[j] = votes[j] + 1
                    break
                #else:
                    #print("Classes " + str(classes[j]) + " diferente de " + str(y_predict[i]))
                    
        # retorna o valor da classe que tem o maior numero de votos
        # TODO: verificar caso quando dá empate
        #print(votes)
        #print(np.argmax(votes))
        #print(classes[np.argmax(votes)])
        return classes[np.argmax(votes)]
    
    # calcula a distancia euclidiana entre a base de dados para cada dado de teste i 
    # e armazena em no atributo dist. Caso o vetor dos dados de teste for diferente, entao
    # repete-se o processo.
    def calculate_and_store_distance(self, X_test):
        #print(KNNClassifier.__X_test)
        #print(X_test)
        if (KNNClassifier.__X_test is np.array(None) or not np.array_equal(KNNClassifier.__X_test, X_test)):
            #print("Salvando array de teste")
            KNNClassifier.__X_test = X_test
            #print(KNNClassifier.__X_test)
            KNNClassifier.__dist = np.zeros((X_train.shape[0], X_test.shape[0]))
           
            for i in range(X_test.shape[0]):
                # clona dado de teste i em todas as linhas
                #print("Calculando distancia do dado de teste " + str(i) + " para toda a base de dados")
                X_new = np.array([X_test[i], ]*X_train.shape[0]) 
                KNNClassifier.__dist[:, i] = np.linalg.norm(X_train - X_new, axis=1)
        else:
            print("Array de distancia já calculado!!!")
            
    # O resultado final é um array com as classes que tiveram o maior numero de votos.
    def predict(self, X_test, classes):
        y_predict = np.zeros((X_test.shape[0],))

        #calcula e armazena as distancia de cada dado de teste a base de dados.
        self.calculate_and_store_distance(X_test)
        
        for i in range(X_test.shape[0]):
            # recupera os index das k menores distancias para o dado de teste i
            #print("Calculando o predição para a entrada X' = " + str(i))
            #set_trace()
            idx = np.argpartition(KNNClassifier.__dist[:, i], self.k)[:self.k]
            y_predict[i] = self.__vote(np.array(y_train)[idx], classes)
            #print(idx, y_predict[i])
        
        return y_predict 


# %% [markdown]
# Classificando usando a classe acima

# %%
#k_neighbors = np.geomspace(1, 1000, 4).astype(int)
#k_neighbors = np.geomspace(1, 74, 1).astype(int)
k_neighbors = np.linspace(1, 80, 81).astype(int)
#_neighbors = np.geomspace(1, 2, 1).astype(int)
# y_teste_min = y_test[0:1000]
# X_test_min = X_test[0:1000]
# X_train_min = X_train[0:1000]
# y_train_min = y_train[0:1000]
classes = np.array(range(0, 6, 1))
y_teste_min = y_test
X_train_min = X_train
y_train_min = y_train
X_test_min = X_test

y_predict = np.zeros((y_teste_min.shape[0], k_neighbors.shape[0]))
for i in range(len(k_neighbors)):
    classifier = KNNClassifier(k_neighbors[i])
#     classifier.fit(X_train_min.values, y_train_min.values)
#     y_predict[:, i] = classifier.predict(X_test_min.values, classes + 1)
    classifier.fit(X_train_min, y_train_min)
    y_predict[:, i] = classifier.predict(X_test_min, classes + 1)

# %% [markdown]
# Calculando a métrica f1_score 

# %%
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# %matplotlib inline

f1_score_array = np.zeros((len(k_neighbors),))
for i in range(len(k_neighbors)):
    f1_score_array[i] = f1_score(y_teste_min,  y_predict[:, i], average='macro')

_= plt.plot(k_neighbors, f1_score_array, '--.')
_= plt.title("F-score evolution");
_= plt.xlabel("threshold - un");
_= plt.ylabel("F-score - un");

index_max = np.argmax(np.array(f1_score_array))
_= plt.plot(k_neighbors[index_max], f1_score_array[index_max], 'X');
_= plt.annotate("Maximum value k-neighbor: " + str(k_neighbors[index_max]) +
                "\nMaximum value F-score: " + str(f1_score_array[index_max]),
            #xy = (matrix_confusion_df['threshold'].iloc[index]/2, f_score.iloc[index]/2))
             xy = (10, 0.86))

print(k_neighbors)
print(f1_score_array)


# %%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

error = []

# Calculating error for K values between 1 and 40
for i in range(len(k_neighbors)):
    knn = KNeighborsClassifier(n_neighbors=k_neighbors[i])
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    f1_score_array_tool_box[i] = f1_score(X_test,  pred_i, average='macro')

_= plt.plot(k_neighbors, f1_score_array_tool_box, '--.')
_= plt.title("F-score evolution tool box");
_= plt.xlabel("threshold - un");
_= plt.ylabel("F-score - un");

index_max = np.argmax(np.array(f1_score_array_tool_box))
_= plt.plot(k_neighbors[index_max], f1_score_array_tool_box[index_max], 'X');
_= plt.annotate("Maximum value k-neighbor: " + str(k_neighbors[index_max]) +
                "\nMaximum value F-score: " + str(f1_score_array_tool_box[index_max]),
            #xy = (matrix_confusion_df['threshold'].iloc[index]/2, f_score.iloc[index]/2))
             xy = (10, 0.86))

print(k_neighbors)
print(f1_score_array_tool_box)
# %%
