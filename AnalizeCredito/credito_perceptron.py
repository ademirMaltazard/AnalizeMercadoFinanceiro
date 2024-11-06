import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


with open('credito_processado.pkl', 'rb') as f:
    x_credito_treino, x_credito_teste, y_credito_treino, y_credito_teste = pickle.load(f)

print(x_credito_teste)

perceptron = Perceptron(max_iter = 2000, tol = 0.001)

perceptron.fit(x_credito_treino, y_credito_treino)

print("Pesos: ", perceptron.coef_)
print("bias: ", perceptron.intercept_)

y_credito_predito = perceptron.predict(x_credito_teste)
acuracia = accuracy_score(y_credito_teste, y_credito_predito)
print('acuracia: ', acuracia)



