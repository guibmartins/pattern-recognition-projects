import pandas as pd, numpy as np
import matplotlib.pyplot as plt, sklearn as skl

from numpy import linalg as la
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

class PCA:
    def __init__(self, k):
        self.__k = k
    
    @property
    def k(self): return self.__k
    
    @k.setter
    def k(self, value): self.__k = value
    
    @property
    def epairs(self): return self.__epairs

    @epairs.setter
    def epairs(self, value): self.__epairs = value

    def decompose(self, X, components=k):
        # calculate the mean for each column
        ux = np.mean(X, axis=0)

        # compute the covariance matrix
        Cov = (X - ux).T @ ((X - ux)) / (X.shape[0] - 1)

        # Eigendecomposition: get the eigevalues 
        # and eigenvectors of the covariance matrix
        evals, evecs = np.linalg.eig(Cov)

        # sorting eigenvectors based on eigenvalues
        self.epairs = [(np.abs(evals[i]), 
                        evecs[:,i]) for i in range(len(evals))]
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

    print('Fitting PCA model...')
    # number of principal components: 2
    model = PCA(2)
    model.decompose(Xstd)
    W, Y = model.transform(Xstd)
    print('PCA: finished.')

    # show PCA results
    classes = ('classe 1', 'classe 2', 'classe 3')
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lbl, col in zip((0, 1, 2), ('indigo', 'orangered', 'deepskyblue')):
            plt.scatter(Y[y==lbl, 0], Y[y==lbl, 1], label=classes[lbl], c=col, alpha=0.9,)
        plt.xlabel('Componente principal 1')
        plt.ylabel('Componente principal 2')
        plt.legend(loc='lower right')
        plt.title('Reducao de dimensionalidade com PCA\nComponentes principais retidos: 2')
        plt.show()

if __name__ == "__main__": main()