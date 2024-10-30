import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

entradas_or = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
saidas_or = np.array([0, 1, 1, 1])

perceptron = Perceptron(max_iter=100, eta0=0.1)
perceptron.fit(entradas_or, saidas_or)

print("Pesos: ", perceptron.coef_)
print("bias: ", perceptron.intercept_)

teste = np.array([[0, 1]])

previsao = perceptron.predict(teste)
print(previsao)

dados = pd.read_csv("data_banknote_authentication.txt", header = None)
x_dados = dados.iloc[:, 0: 4].values
y_dados = dados.iloc[:, 4].values

perceptron.fit(x_dados, y_dados)

print("Pesos: ", perceptron.coef_)
print("bias: ", perceptron.intercept_)

previsao = perceptron.predict(x_dados)
acuracia = accuracy_score(y_dados, previsao)

print(acuracia)