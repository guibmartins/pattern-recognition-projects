import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron(object):
    def __init__(self, theta=0, lrate=0.01, epochs=100):
        self.theta = theta
        self.lrate = lrate
        self.epochs = epochs
    
    def signal(self, x):
        return 1 if x >= self.theta else 0

    # training function
    def fit(self, X, d, theta=None, lrate=None, epochs=None):
        
        if theta is None: theta = self.theta
        if lrate is None: lrate = self.lrate
        if epochs is None: epochs = self.epochs
        
        bias = np.ones((X.shape[0], 1), dtype=float)
        X = np.concatenate((bias, X), axis = 1)
        
        w = np.random.random_sample(size=(X.shape[1]))
        g = np.zeros(d.shape)
        y = np.zeros(d.shape)

        t = 0
        cost = []
        error_treshold = 0.001
        e = 1.0
        while t < epochs and e > error_treshold:
            e = 0
            for i in range(X.shape[0]):
                # computing output (signal function)
                y[i] = self.signal(w @ X[i])

                # updating weights and computing training error
                w += lrate * (d[i] - y[i]) * X[i]
                e += abs(d[i] - y[i])
                
            e /= X.shape[0]
            cost.append(e)
            t += 1

        self.epochs = t
        return w, cost


    def predict(self, w, X): 
        pass


def main():

    features, labels = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=18)
    print("Amostras: " , features.shape)
    print("Classes: ", np.unique(labels))

    figure, subfig = plt.subplots(1, 1, figsize=(5, 5))
    subfig.scatter(features[:, 0], features[:, 1], c=labels)
    subfig.set_title('ground truth', fontsize=20)
    plt.show()

    X = np.array(features)
    d = np.array(labels)

    model = Perceptron(lrate=0.001, epochs=200)
    w, cost = model.fit(X, d)

    print('Total de épocas necessárias para o aprendizado: ', model.epochs)
    print('Pesos aprendidos: ', w)

    # plot the data
    fig = plt.figure()
    plt.plot(range(1, len(cost)+1), cost, color='purple')
    plt.grid(True)
    plt.xlabel('# épocas')
    plt.ylabel('Erro absoluto)')
    plt.title('Custo de treinamento em função do número de épocas')
    plt.show()


if __name__ == "__main__": main()


