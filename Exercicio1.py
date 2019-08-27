#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:34:42 2019

@author: vitor
"""
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
base = pd.read_csv('/home/vitor/Downloads/Tabela.csv', sep= ';', encoding='utf8')
print(base.columns)

previsor = base['texto_port']
classe = base['sentimento']
print(previsor.shape)
print(classe.shape)

vetor_palavras = CountVectorizer(analyzer="word")
frequencia_palavra = vetor_palavras.fit_transform(previsor)
modelo = MultinomialNB()
modelo.fit(frequencia_palavra,classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(frequencia_palavra, classe, test_size=0.3, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)# n_neighbors (k) vizinhos, metrica minkwski, p=2 é medida euclidiana padrão
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)


print(previsoes)
print(matriz)
print(precisao)
print(previsoes)