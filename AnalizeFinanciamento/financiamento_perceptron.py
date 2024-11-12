import pickle
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


with open('financiamento_processado.pkl', 'rb') as f:
    x_financiamento_treino, y_financiamento_treino, x_financiamento_teste, y_financiamento_teste = pickle.load(f) 

print('X treino:\n', x_financiamento_treino)
print('X teste:\n', x_financiamento_teste)
print('Y teste:\n', y_financiamento_treino)
print('Y treino:\n', y_financiamento_teste)

perceptron = Perceptron(max_iter=2000, tol=0.05)

print('treino perceptron')
perceptron.fit(x_financiamento_treino, y_financiamento_treino)

print('Pesos: ', perceptron.coef_)
print('Bias: ', perceptron.intercept_)

y_predito = perceptron.predict(x_financiamento_teste)
acuracia = accuracy_score(y_financiamento_teste, y_predito)

print('Acuracia: ', acuracia)

