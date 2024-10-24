# CRIAÇÃO DO NEURONIOS PARA TABELAS VERDADES

import numpy as np
import pandas as pd

# FUNÇÃO DE ATIVAÇÃO
def funcaoDegrau(valor):
    if valor <= 0:
        return 1
    else:
        return 0

# FUNÇÃO DE PREVISÃO
def prever(entrada, pesos, bias):
    soma = np.dot(entrada, pesos) + bias
    resultado = funcaoDegrau(soma)
    return resultado

# TREINAMENTO PERCEPTRON
def treinoPerceptron(entrada_treino, saida, taxa_aprendizagem = 0.1, max_epocas = 100):
    pesos = np.zeros(entrada_treino.shape[1])
    bias = 0

    for i in range(max_epocas):
        for entrada, rotulo in zip(entrada_treino, saida):
            previsao = prever(entrada, pesos, bias)
            erro = rotulo - previsao
            pesos += taxa_aprendizagem * erro * entrada
            bias += taxa_aprendizagem * erro

    return 0

dados_or = pd.DataFrame({
    'entrada1': [0, 0, 1, 1],
    'entrada2': [0, 1, 0, 1],
    'saida': [0, 1, 1, 1]
})

print(dados_or)

entradas_or = dados_or[['entrada1', 'entrada2']].values
saida_or = dados_or['saida'].values

print(entradas_or, '\n', saida_or)
