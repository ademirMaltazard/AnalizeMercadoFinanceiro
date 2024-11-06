import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

base_credito = pd.read_csv('AnalizeCredito/credit_data.csv')
print(base_credito)
print(base_credito.describe())

print(base_credito[base_credito['age'] < 0])
'''
# GRAFICO DE CONTAGEM
plt.figure(1)
sns.countplot(x = base_credito['age'])

# GERAR HISTOGRAMA
plt.figure(2)
plt.hist(x = base_credito['income'])
plt.show()

# GRAFICOS DINAMICOS
grafico = px.scatter_matrix(base_credito, dimensions=['age', 'income'])
grafico.show()
'''


# TRATAMENTO DOS DADOS
# primeira opção: DELETAR TODA A COLUNA
base_credito2 = base_credito
base_credito2 = base_credito2.drop('age', axis = 1)
print(base_credito2)

# segunda opção: DELETAR APENAS AS LINHAS COM PROBLEMAS
base_credito3 = base_credito
base_credito3 = base_credito3.drop(base_credito3[base_credito3['age'] < 0].index)
print(base_credito3)

# REMOVER DADOS FALTANTES
## encontrar nulos
print(base_credito3['age'].isnull())
## deletar nulos
base_credito3 = base_credito3.dropna()
print(base_credito3.loc[pd.isnull(base_credito3['age'])])


# terceira opção: SUBSTITUIR PELA MÉDIA
base_credito.loc[base_credito['age'] < 0, 'age'] = base_credito3['age'].mean()
base_credito = base_credito.fillna(base_credito3['age'].mean())
print(base_credito.describe())
print('Nulos: ', base_credito['age'].isnull().sum())

# APLICAR LABEL ENCODER
## não necessita pois n tem indices categóricos

# SEPARAÇAO DAS CLASSES E PREVISORES
x_credito = base_credito.iloc[:, 1: 4].values
y_credito = base_credito.iloc[:, 4].values

print(x_credito)
print(y_credito)

# NORMALIZAÇAO - intervalo de [0,1]
from sklearn.preprocessing import MinMaxScaler

normalizacao = MinMaxScaler()
x_credito = normalizacao.fit_transform(x_credito)
print('Normalizado\n', x_credito)

# PADRONIZAÇAO - [média == 0, desvio padrão == 1] - distribuição normal



# SEPARAÇAO DE TREINO E TESTE
from sklearn.model_selection import train_test_split

x_credito_treino, x_credito_teste, y_credito_treino, y_credito_teste = train_test_split(x_credito, y_credito, test_size = 0.25, random_state = 0)

# SALVAR DADOS
import pickle

with open('credito_processado.pkl', 'wb') as f:
    pickle.dump((x_credito_treino, x_credito_teste, y_credito_treino, y_credito_teste), f)
    print('Dados salvos com sucesso!')








