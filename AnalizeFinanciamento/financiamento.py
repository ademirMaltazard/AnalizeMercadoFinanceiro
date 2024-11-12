import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# carregar dados
financiamento_data = pd.read_csv('AnalizeFinanciamento/financial_data.csv')

print(financiamento_data)

'''
# verificar outliers
plt.figure()
grafico = px.scatter_matrix(financiamento_data, dimensions=['Age', 'Account_Type'])
grafico.show()
'''

# verificar nulos
print('Verificando nulos:')
print(financiamento_data.isnull().sum())

# correção de dados nulos
print('Removendo nulos para Calculo de média:')
financiamento_data_1 = financiamento_data.dropna()
print(financiamento_data_1.isnull().sum())

# preenchendo Nulos com media do existentes
# Coluna Age
print('preenchendo nulos na idade')
financiamento_data["Age"] = financiamento_data["Age"].fillna(financiamento_data_1['Age'].mean())
print(financiamento_data.isnull().sum())
# coluna Credit_Score
print('preenchendo nulos no score')
financiamento_data["Credit_Score"] = financiamento_data["Credit_Score"].fillna(financiamento_data_1['Credit_Score'].mean())
print(financiamento_data.isnull().sum())

# separar previsores e classe
x_financiamento = financiamento_data.iloc[:, 0:4].values
y_financiamento = financiamento_data.iloc[:, 4].values

print(x_financiamento)
print(y_financiamento)

# aplicar label encoder
lb_AnnualIncome = LabelEncoder()
lb_AccountType = LabelEncoder()

x_financiamento[:, 1] = lb_AnnualIncome.fit_transform(x_financiamento[:, 1])
x_financiamento[:, 2] = lb_AccountType.fit_transform(x_financiamento[:, 2])

# criar OneHotEncoder
oneHotEncoder = ColumnTransformer(
    transformers = [('OneHot', OneHotEncoder(sparse_output=False), [1, 2])],
    remainder = 'passthrough'
)

x_financiamento = oneHotEncoder.fit_transform(x_financiamento)

print('depois do OneHotEncoder')
print(x_financiamento)

# Normalizar dados
normalizador = MinMaxScaler()
x_financiamento = normalizador.fit_transform(x_financiamento)
print('Dado Normalizado:\n', x_financiamento)

# separar treino e teste
x_financiamento_treino, x_financiamento_teste, y_financiamento_treino, y_financiamento_teste = train_test_split(x_financiamento, y_financiamento, test_size= 0.25, random_state= 0)

# salvar em arquivo
with open('financiamento_processado.pkl', 'wb') as f:
    pickle.dump([x_financiamento_treino, y_financiamento_treino, x_financiamento_teste, y_financiamento_teste], f)
    print('Arquivo criado com sucesso!')









