# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.metrics import confusion_matrix

# +
dados_teste = sio.loadmat('dados_teste.mat')
dados_treino = sio.loadmat('dados_treinamento.mat')
dados_valid = sio.loadmat('dados_val.mat')

x_teste = np.array(dados_teste["Xt"])
y_teste =  np.array(dados_teste["yt"])
x_treino =  np.array(dados_treino["X"])
y_treino =  np.array(dados_treino["y"])
x_valid =  np.array(dados_valid["Xval"])
y_valid =  np.array(dados_valid["yval"])


#
teste_df=pd.DataFrame(x_teste,columns=['x1','x2'])
teste_df['y']=y_teste
treino_df=pd.DataFrame(x_treino,columns=['x1','x2'])
treino_df['y']=y_treino
valid_df=pd.DataFrame(x_valid,columns=['x1','x2'])
valid_df['y']=y_valid
total_df=pd.DataFrame(x_teste,columns=['x1','x2'])
total_df['y']=y_teste
total_df.append(treino_df, ignore_index=True)
total_df.append(valid_df, ignore_index=True)

print(teste_df.describe())
print(treino_df.describe())
print(valid_df.describe())
print(total_df.describe())
print(total_df.hist())


# +
def labelizer(input,positive_label,negative_label):
    input_intern=input
    for i in range(0,len(input)):
        if input[i]==positive_label:
            input_intern[i]=1
            
        if input[i]==negative_label:
            input_intern[i]=0            
    return input_intern

y_teste_rn=labelizer(y_teste,1,-1)
y_treino_rn=labelizer(y_treino,1,-1)
y_valid_rn=labelizer(y_valid,1,-1)


# +
def cria_mlp(neuronios,otimizador='Adam',ativacao='relu'):
    rede_neural=Sequential()
    rede_neural.add(Dense(neuronios, input_dim=2, activation=ativacao))
    rede_neural.add(Dense(1, activation='sigmoid'))
    # Para entropia cruzada
    rede_neural.compile(loss='binary_crossentropy', optimizer=otimizador, metrics=['accuracy'])
    # Para o erro quadrático médio
#     rede_neural.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['accuracy'])
    return rede_neural

def mostra_grafico_custo(historico):
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.title('Custo')
    plt.ylabel('Custo')
    plt.xlabel('Epoca')
    plt.yscale("log")
    plt.legend(['Treino', 'Validacao'], loc='upper left')
    plt.show()
def mostra_grafico_accuracy(historico):
    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    plt.title('Acurácia')
    plt.ylabel('Acurácia')
    plt.xlabel('Epoca')
    plt.yscale("log")
    plt.legend(['Treino', 'Validacao'], loc='upper left')
    plt.show()
def mostra_mensagem(historico):
    print("Custo final treino:")
    print(historico.history['loss'][-1])
    print("Custo final validação:")
    print(historico.history['val_loss'][-1])
    print("Acurácia final treino:")
    print(historico.history['accuracy'][-1])
    print("Acurácia final validação:")
    print(historico.history['val_accuracy'][-1])
def plot_decision_boundary(X, y, model, steps=100, cmap='Paired'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    colors = ['blue' if label == 1 else 'red' for label in y]
    ax.scatter(X[:,0], X[:,1], s=7, c=colors, cmap=cmap, lw=0)

    return fig, ax

def aplica_threshold(y,threshold):
    y_temp=y.copy()
    for i in range(0,len(y)):
        if y[i]>=threshold:
            y_temp[i]=1
        else:
            y_temp[i]=0
    return y_temp



# +
# Letra a
epocas=5000
tamanho_batch=100
neuronios=40
#Teste 1 ReLu Adam
print("Teste 1: Relu Adam")
rede_neural=cria_mlp(neuronios,otimizador='Adam',ativacao='relu')
#Rede com as labels alteradas
historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch, verbose=0)
mostra_grafico_custo(historico)
mostra_grafico_accuracy(historico)
mostra_mensagem(historico)

#Teste 2 Relu Sgd
print("Teste 2: Relu Sgd")
rede_neural = cria_mlp(neuronios,otimizador='sgd',ativacao='relu')
#Rede com as labels alteradas
historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch, verbose=0)
mostra_grafico_custo(historico)
mostra_grafico_accuracy(historico)
mostra_mensagem(historico)

#Teste 3 Logistica Adam
print("Teste 3: Logistica Adam")
rede_neural=cria_mlp(neuronios,otimizador='Adam',ativacao='sigmoid')
#Rede com as labels alteradas
historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch, verbose=0)
mostra_grafico_custo(historico)
mostra_grafico_accuracy(historico)
print(max(historico.history['val_loss']))
mostra_mensagem(historico)

#Teste 2 Logistica Sgd
print("Teste 4: Logistica Sgd")
rede_neural = cria_mlp(neuronios,otimizador='sgd',ativacao='sigmoid')
#Rede com as labels alteradas
historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch, verbose=0)
mostra_grafico_custo(historico)
mostra_grafico_accuracy(historico)
mostra_mensagem(historico)

# +
# Letra b
epocas=5000
tamanho_batch=100
rede_neural = cria_mlp(40,otimizador='sgd',ativacao='relu')
#Rede com as labels alteradas
historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch, verbose=0)

plot_decision_boundary(x_treino,y_treino_rn,rede_neural,steps=1000,cmap='RdBu')

# -

#Letra c
print(rede_neural.evaluate(x_teste, y_teste_rn))
y=rede_neural.predict(x_teste)
print(confusion_matrix(aplica_threshold(y,0.5),y_teste_rn))
tn, fp, fn, tp = confusion_matrix(aplica_threshold(y,0.5),y_teste_rn).ravel()
print("Erro Percentual:")
print((fp+fn)/(tn+fp+fn+tp))

neuronios=[10,20,40,80,160,320]
# neuronios=['10','20']
epocas=5000
tamanho_batch=200
#ENTENDER MELHOR COMO ESSA VARIAÇÃO INFLUENCIA A PARADA
variacao=0.00001
paciencia=200
relatorio=pd.DataFrame(columns=['Neuronios','Acurácia Treino','Acurácia Validação'])
k=-1
for i in neuronios:
    k+=1
    print("Numero Neuronios")
    print(i)
    relatorio.loc[k,'Neuronios']=i
    rede_neural = cria_mlp(int(i),otimizador='sgd',ativacao='relu')
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=variacao,patience=paciencia, verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    historico=rede_neural.fit(x_treino, y_treino_rn,validation_data=(x_valid,y_valid_rn), epochs=epocas, batch_size=tamanho_batch,callbacks=[es,mc], verbose=0)
    rede_neural=load_model('best_model.h5')
#     O primeiro valor é o custo o segundo a acurácia
    relatorio.loc[k,'Acurácia Treino']=rede_neural.evaluate(x_treino,y_treino_rn,verbose=0)[1]
    relatorio.loc[k,'Acurácia Validação']=rede_neural.evaluate(x_valid,y_valid_rn,verbose=0)[1]
print(relatorio)

# +
# #Teste, após este teste vimos que cada neurônio implementa o bias e de fato temos que colocar as lavbels corretamente 0,1.

# teste=Sequential()
# teste.add(Dense(1, input_dim=1, activation='sigmoid'))
# teste.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# x=np.array([0,1])
# y=np.array([-1,1])
# historico=teste.fit(x,y, epochs=100, batch_size=10)
# print(teste.predict([0,1]))
# for layer in teste.layers:
#     print(layer.get_weights())
