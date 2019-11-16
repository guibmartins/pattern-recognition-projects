import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

from numpy import linalg as la
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

class LDA:
    def __init__(self): pass

    @property
    def epairs(self): return self.__epairs

    @epairs.setter
    def epairs(self, value): self.__epairs = value
    
    def decompose(self, ds, X):
        y = ds.target

        Ux = []
        for c in range(ds.target_names.shape[0]):
            Ux.append(np.mean(X[y==c], axis=0))
        
        d = X.shape[1]
        Sw = np.zeros((d,d))

        for c, ux in zip(range(1,d), Ux):
            Sc = np.zeros((d,d))                  
            for row in X[y == (c-1)]:
                row, ux = row.reshape(d,1), ux.reshape(d,1)
                Sc += (row - ux).dot((row - ux).T)
            Sw += Sc

        u = np.mean(X, axis=0)
        Sb = np.zeros((d,d))
        for i, ux in enumerate(Ux):  
            n = X[y==i,:].shape[0]
            ux = ux.reshape(d,1)
            u = u.reshape(d,1)
            Sb += n * (ux - u).dot((ux - u).T)
        
        # eigendecomposition of (Sw-¹ * Sb) 
        evals, evecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        # sorting eigenvectors based on eigenvalues
        self.epairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]
        self.epairs.sort(key=lambda x: x[0], reverse=True) 

    def transform(self, X):
        W = np.hstack((self.epairs[0][1].reshape(X.shape[1],1), 
                        self.epairs[1][1].reshape(X.shape[1],1)))
        Y = X @ W
        return [W,Y]

def main():
    # loading dataset 'wine'
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    # standardizing features
    Xstd = skl.preprocessing.StandardScaler().fit_transform(X)

    print('Fitting LDA model...')
    model = LDA()
    model.decompose(wine, Xstd)
    W, Y = model.transform(Xstd)
    print('LDA: finished.')

    # show LDA results
    classes = ('classe 1', 'classe 2', 'classe 3')
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for c,marker,color in zip(range(wine.target_names.shape[0]),
                                    ('s', 'x', 'o'), ('indigo', 'orangered', 'deepskyblue')):
            plt.scatter(x=Y[:,0].real[y == c], y=Y[:,1].real[y == c],
                    marker=marker, color=color, alpha=0.9, label=classes[c])
        plt.xlabel('Discriminante linear 1')
        plt.ylabel('Discriminante linear 2')
        plt.legend(loc='upper right')
        plt.title('Reducao de dimensionalidade com LDA\nProjeção nos dois primeiros discriminantes lineares')
        plt.show()

if __name__ == "__main__": main()
