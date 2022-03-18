"""O objetivo é localizar o local onde tenha ocorrido um
curto circuito em linha de transmissão curta,
média ou longa (centenas de Km)"""

"""Exploração da base de dados, a BD está especificado em CSV mas poderia ser em outros formatos.

Visualização da base de dados.

Verificação se existe lacunas na base de dados (Dados faltantes).

Depois de saber sobre os valores faltantes na base de dados, acha-se a 
média, essa média é que irá preecheer os espaços em vazios na BD.

Mostrar somente as linhas com os dados faltantes.

Preenchimento dos dados faltantes na BD.

Nessa etapa será mostrado os trechos que estavam em faltas, e agora ja preenchidos.

Sendo uma rede supervisionada o que quer dizer que tem-se a refência, com isso o "x_faltas" será o Previsor.

visualizar os valores x_faltas

O "y_faltas" representa os valores de referência.

Como normalmente os valores de entrada para treinamentos e testes podem estar defasado, 
(uns muito altos e outro e outros muito baixo), 
quando isso acontece, afeta negativamente nos valores de saida, por esse motivo é fundamental deixar os valores na mesma escala.

Visualizar colocado na mesma escala.

75% dos dados serão para treinamento, 25% dos dados para testes."""

import pandas as pd
import numpy as np

base_faltas = pd.read_csv("Base_de_dados")
base_faltas
base_faltas.isnull().sum()
base_faltas["trecho em falta"].mean()
base_faltas["trecho em falta"].mean()
base_faltas.loc[pd.isnull(base_faltas["trecho em falta"])]
base_faltas.loc[pd.isnull(base_faltas["trecho em falta"])]
base_faltas["trecho em falta"].fillna(base_faltas["trecho em falta"].mean(), inplace=True)
base_faltas["trecho em falta"].fillna(base_faltas["trecho em falta"].mean(), inplace=True)
base_faltas.loc[pd.isnull(base_faltas["agora preenchido"])]
base_faltas.loc[pd.isnull(base_faltas["agora preenchido"])]

x_faltas = base_faltas.iloc[previsores sem os valores de referencia].values
x_faltas
y_faltas = base_faltas.iloc[valores que pretendemos alcançar].values
y_faltas

from sklearn.preprocessing import StandardScaler
scalar_faltas = StandardScaler()
x_faltas = scalar_faltas.fit_transform(x_faltas)
x_faltas

"""Criar variaveis"""
from sklearn.model_selection import train_test_split
x_faltas_treinamento, x_faltas_teste, y_faltas_treinamento, y_faltas_teste = train_test_split(x_faltas,y_faltas, test_size=0.25, random_state=0)

x_faltas_treinamento.shape
y_faltas_treinamento.shape
x_faltas_teste.shape
y_faltas_teste.shape

"""Guardando variaveis"""

import pickle
with open("faltas.pkl", mode = "wb") as f:
  pickle.dump([x_faltas_treinamento,y_faltas_treinamento,x_faltas_teste,y_faltas_teste], f)

"""Aplicando Tecnicas De Redes Neurais Artificiais"""

from sklearn.neural_network import MLPClassifier
import pickle
with open("faltas.pkl", "rb") as f:
  x_faltas_treinamento,y_faltas_treinamento,x_faltas_teste,y_faltas_teste = pickle.load(f)

x_faltas_treinamento.shape, y_faltas_treinamento.shape
x_faltas_teste.shape, y_faltas_teste.shape

rede_neural_faltas = MLPClassifier(max_iter=1000, verbose=True,tol=0.0000100, solver = "adam", activation='relu', hidden_layer_sizes=(2,2))
rede_neural_faltas.fit(x_faltas_treinamento, y_faltas_treinamento)

previsoes = rede_neural_faltas.predict(x_faltas_teste)
previsoes

y_faltas_teste

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_faltas_teste, previsoes)

print(classification_report(y_faltas_teste, previsoes))
