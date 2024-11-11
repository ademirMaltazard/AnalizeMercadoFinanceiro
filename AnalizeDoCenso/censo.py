import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

base_censo = pd.read_csv('AnalizeDoCenso/census.csv')
print(base_censo)

print(base_censo.describe())

x_censo = base_censo.iloc[:, 0: 14].values
y_censo = base_censo.iloc[:, 14].values

print(x_censo)
print(y_censo)

# LabelEncoder
from sklearn.preprocessing import LabelEncoder

lb_workclass = LabelEncoder()
lb_education = LabelEncoder()
lb_maritial_status = LabelEncoder()
lb_ocupation = LabelEncoder()
lb_relationship = LabelEncoder()
lb_race = LabelEncoder()
lb_sex = LabelEncoder()
lb_nativeCountry = LabelEncoder()
lb_income = LabelEncoder()

x_censo[:, 1] = lb_workclass.fit_transform(x_censo[:, 1])
x_censo[:, 3] = lb_education.fit_transform(x_censo[:, 3])
x_censo[:, 5] = lb_maritial_status.fit_transform(x_censo[:, 5])
x_censo[:, 6] = lb_ocupation.fit_transform(x_censo[:, 6])
x_censo[:, 7] = lb_relationship.fit_transform(x_censo[:, 7])
x_censo[:, 8] = lb_race.fit_transform(x_censo[:, 8])
x_censo[:, 9] = lb_sex.fit_transform(x_censo[:, 9])
x_censo[:, 13] = lb_nativeCountry.fit_transform(x_censo[:, 13])
y_censo[:] = lb_income.fit_transform(y_censo[:])

print(x_censo)
print(y_censo)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Criar o OneHotEncoder para as colunas especificas

cod_OneHotEncoder = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(sparse_output=False),[1,3,5,6,7,8,9,13])], 
    remainder='passthrough'
    )

x_censo = cod_OneHotEncoder.fit_transform(x_censo)

print('Depois de onehot\n')
print(x_censo)

# transformar para ARRAY
x_censo = x_censo.toarray()

# Mormalizar os dados
from sklearn.preprocessing import MinMaxScaler

minMaxScaler = MinMaxScaler()
x_censo = minMaxScaler.fit_transform(x_censo)
print(x_censo)

#separação de treino e teste
from sklearn.model_selection import train_test_split

x_censo_Treino, x_censo_teste, y_censo_treino, y_censo_teste = train_test_split(x_censo, y_censo, test_size = .25, random_state = 0)

# salvar em arquivo
import pickle

with open('censo_processado.pkl', 'wb') as f:
    pickle.dump([x_censo_teste, y_censo_teste, x_censo_Treino, y_censo_treino], f)
    print("Arquivo criado com sucesso!")





