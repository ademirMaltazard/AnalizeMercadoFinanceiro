import pickle
from sklearn.linear_model import Perceptron

with open('censo_processado.pkl', 'rb') as f:
    x_censo_teste, y_censo_teste, x_censo_Treino, y_censo_treino = pickle.load(f)

print(x_censo_Treino)
perceptron = Perceptron(max_iter=2000, tol= 0.05)
perceptron.fit(x_censo_Treino, y_censo_treino)

print('Treino')