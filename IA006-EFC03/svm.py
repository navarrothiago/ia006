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

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import scipy.io as sio
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn import preprocessing
# %matplotlib inline

# %%
dados_teste = sio.loadmat('dados_teste.mat')
dados_treino = sio.loadmat('dados_treinamento.mat')
dados_valid = sio.loadmat('dados_val.mat')

X_test = np.array(dados_teste["Xt"])
y_test =  np.array(dados_teste["yt"]).astype(np.float)
X_train =  np.array(dados_treino["X"])
y_train =  np.array(dados_treino["y"]).astype(np.float)
X_valid =  np.array(dados_valid["Xval"])
y_valid =  np.array(dados_valid["yval"]).astype(np.float)

print(np.mean(X_train, axis=0))


# %%
print(np.mean(X_train, axis=0))
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
print(scaler.mean_)
print(scaler.var_)

# %%
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(y_train, bins=20, color='#0504aa',
                            alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y_test')
plt.ylabel('Frequency')
plt.title('Y Train Histogram')
plt.show()

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(X_train, bins=20,alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('X_train')
plt.ylabel('Frequency')
plt.title('Y train Histogram')
plt.show()


# %%
    
def svm_fit_plot_report(X_train, y_train, X_valid, y_valid, gamma_array, penality_array):
    scores = ['accuracy']    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        parameters = {'C':penality_array, 'gamma':gamma_array}

        # The indices which have the value -1 will be kept in train.
        train_indices = np.full((len(X_train),), -1, dtype=int)

        # The indices which have zero or positive values, will be kept in test
        test_indices = np.full((len(X_valid),), 0, dtype=int)
        test_fold = np.append(train_indices, test_indices) 

        ps = PredefinedSplit(test_fold)
        
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, cv=ps, n_jobs=-1)
        
        X = np.concatenate((X_train, X_valid))
        y = np.concatenate((y_train.ravel(), y_valid.ravel()))
        print(len(X_train))
        print(len(X))
        clf.fit(X, y)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        h = .02  # step size in the mesh
        
        x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
        y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.bwr)

        plt.title('2-Class classification using Support Vector Machine')
        plt.axis('tight')
        plt.show()


# %%
svm_fit_plot_report(X_train, y_train.ravel(),
                       X_valid, y_valid.ravel(),
                       np.geomspace(0.001, 1000, 7), #gamma
                       np.geomspace(0.001, 1000, 7)) #penality

# %%
