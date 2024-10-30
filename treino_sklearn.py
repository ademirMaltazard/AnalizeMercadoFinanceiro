import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

entradas_or = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
saidas_or = np.array([0, 1, 1, 1])

perceptron = Perceptron(max_iter=100, eta0=0.1)
perceptron.fit(entradas_or, saidas_or)
print('+++')

print("Pesos: ", perceptron.coef_)
print("bias: ", perceptron.intercept_)

teste = np.array([[0, 1]])

previsao = perceptron.predict(teste)
print(previsao)
