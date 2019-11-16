import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

from numpy import linalg as la
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA


class KPCA:
    def __init__(self, k, kernel='rbf', gamma=1):
        self.__k = k
        self.__kernel = kernel
        self.__gamma = gamma
    
    @property
    def k(self): return self.__k
    
    @k.setter
    def k(self, value): self.__k = value
    
    @property
    def kernel(self): return self.__kernel
    
    @kernel.setter
    def kernel(self, value): self.__kernel = value
    
    @property
    def gamma(self): return self.__gamma
    
    @gamma.setter
    def gamma(self, value): self.__gamma = value

    def compute_kernelmatrix(self, X):
        S = []
        K_0 =[]

        if self.kernel == 'poly': 
            # polynomial kernel
            K_0 = (X @ X.T) ** self.gamma
        else: 
            # gaussian radial basis function (rbf) kernel
            S = squareform(pdist(X, 'sqeuclidean'))
            K_0 = exp(-self.gamma * S)

        # dimensionality of K
        N = K_0.shape[0]

        # get a all-one / N NxN matrix 
        On = np.ones((N,N)) / N
        return ( K_0 - (On @ K_0) - (K_0 @ On) + ((On @ K_0) @ On) )

    def decompose(self, K, components=k):
        # Eigendecomposition: get the eigevalues 
        # and eigenvectors of the covariance matrix
        evals, evecs = np.linalg.eigh(K)

        # sorting eigenvectors based on eigenvalues
        X_kpca = np.column_stack((evecs[:,-i] for i in range(1, components + 1)))
        return X_kpca

def main():
    # dataset: make circles
    X, y = make_circles(n_samples=1000, factor=.3, noise=.05)
    Xstd = StandardScaler().fit_transform(X)

    print('Fitting KPCA model...')
    model = KPCA(2, kernel='rbf', gamma=0.5)
    K = model.compute_kernelmatrix(Xstd)
    X_kpca = model.decompose(K, components=2)
    print('done.')

    classes = ('classe 1', 'classe 2')
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8,6))
        for lbl, col in zip((0,1), ('orangered','deepskyblue')):
            plt.scatter(X_kpca[y==lbl, 0], X_kpca[y==lbl, 1], 
                        label=classes[lbl], c=col, alpha=0.9,)
        plt.xlabel('Componente principal 1')
        plt.ylabel('Componente principal 2')
        plt.legend(loc='upper right')
        plt.title('Reducao de dimensionalidade com Kernel PCA\nComponentes principais retidos: 2\nGamma={}'.format(model.gamma))
        plt.show()

if __name__ == "__main__": main()
