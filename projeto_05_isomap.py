import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.manifold import isomap

class ISOMAP:

    def __init__(self, neighbors=5, dim=2):
        self.__neighbors = neighbors
        self.__dim = dim
    
    @property
    def dim(self): return self.__dim

    @dim.setter
    def dim(self, value): self.__dim = value

    @property
    def neighbors(self): return self.__neighbors

    @neighbors.setter
    def neighbors(self, value): self.__neighbors = value


    def fit(self, X, neighbors=None, dim=None):
        if neighbors is None: neighbors = self.neighbors
        if dim is None: dim = self.dim

        # computing the knn graph of input X
        N = isomap.kneighbors_graph(X, neighbors, mode='distance')
        D = isomap.graph_shortest_path(N, method='D', directed=False)
        n = D.shape
        
        A = -0.5 * D
        I = np.identity(n[0], dtype=float)
        U = np.ones(n, dtype=float)
        H = I - ((1 / n[0]) * U)
        B = H @ A @ H

        # finding eigenvalues and eigenvectors
        eval, evec = np.linalg.eigh(B)
        
        # constructing eigenmatrices of m-eigenvalues and m-eigenvectors
        V_hat = np.column_stack((evec[:,-i] for i in range(1, dim + 1)))
        
        L = []
        for i in range(1, dim + 1): L.append(eval[-i])
        L_hat = np.diag(np.array(L))
    
        # mapping points (embeddings)
        X_hat = (L_hat ** 0.5) @ V_hat.T

        return X_hat

def main():
    # loading dataset
    X, color = make_swiss_roll(n_samples=1200, random_state=123)
    Xstd = StandardScaler().fit_transform(X)

    print('Fitting ISOMAP model...')
    model = ISOMAP(neighbors=9, dim=2)
    X_hat = model.fit(Xstd)
    print('done.')

    # show ISOMAP results
    Y = X_hat.T
    plt.figure(figsize=(8,6))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.rainbow)
    plt.title('Aprendizado de variedades com ISOMAP\nVizinhos-mais-próximos: {}'.format(model.neighbors))
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.show()

if __name__ == "__main__": main()
