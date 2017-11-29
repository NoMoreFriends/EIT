#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed May 31 13:14:36 2017

@author: ismailerradi
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier as rf
from scipy.sparse import *
from sklearn.svm import LinearSVC as LSVC
import pandas as pd
import numpy as np
import time
import unicodedata as ud
import nltk as n
import enchant
from spellcheck import SpellingReplacer

columns = ['docID', 'year', 'gender', 'age', 'lod', 'lineID', 'rawText', 'intT', 'intV', 'Cause', 'stdText', 'ICD10']
DEL = ['docID', 'year', 'gender', 'age', 'lod', 'lineID', 'intT', 'intV', 'Cause', 'stdText']


def save_sparse(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse(filename):
    loader = np.load(filename)
    return csc_matrix(( loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def normalize(s):
    s = ud.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return s.lower()

def tokens(line):
    words = n.RegexpTokenizer(r'\w+').tokenize(line)
    return words

def tokens_corr(line):
    words= n.RegexpTokenizer(r'\w+').tokenize(line)
    for i in range(len(words)):
        words[i] = correction.replace(words[i])
    return words
        

def fill_map(df):
    j = 0
    for i,d in enumerate(df['rawText']):
        words = tokens_corr(d)
        df['rawText'][i] = ' '.join(words)
        for k in words:
            if k not in hashmap:
                hashmap[k] = j
                j = j+1


def model_class(set):
    df = set['rawText']
    x = []
    y = []
    ones = []
    for i in range(len(set)):
        words = tokens(df[i])
        for k in words:
            x.append(i)
            y.append(cols.index(k))
            ones.append(1)
    x = np.asarray(x)
    y = np.asarray(y)
    ones = np.asarray(ones)
    X = csc_matrix( (ones, (x, y)), shape=(len(set), len(cols)))
    return X


data = pd.read_csv(
    '/root/Bureau/EIT/corpus/train/train.csv',
    skipinitialspace=True,
    sep=';',
    names=columns)

dev = pd.read_csv(
    '/root/Bureau/EIT/corpus/dev/dev_full.csv',
    skipinitialspace=True,
    sep=';',
    names=columns)

data = data.dropna(subset=['ICD10'])
data = data.reset_index(drop=True)
dev = dev.dropna(subset=['ICD10'])
dev = dev.reset_index(drop=True)

for i in DEL:
    data.drop(i, axis=1, inplace=True)
    dev.drop(i, axis=1, inplace=True)

data = data[:30000]
dev = dev[:30000]

data['rawText'] = data['rawText'].apply(normalize)
dev['rawText'] = dev['rawText'].apply(normalize)

"""hashmap = {}
correction = SpellingReplacer()
fill_map(data)
fill_map(dev)
cols = list(hashmap.keys())
print(cols)
print(len(cols))
trainX = model_class(data)
evalX = model_class(dev)
save_sparse('/root/Bureau/train_spell30', trainX)
save_sparse('/root/Bureau/eval_spell30', evalX)
"""
trainX = load_sparse('/root/Bureau/train_spell30.npz')
evalX = load_sparse('/root/Bureau/eval_spell30.npz')
trainY = data['ICD10'].values
evalY = dev['ICD10'].values
print("trainX :", trainX.shape)
print("trainY :", trainY.shape)
print("evalX :", evalX.shape)
print("evalY :", evalY.shape)

models = []
models.append(('rf', rf(n_estimators=10)))
models.append(('LSVC', LSVC(random_state=0)))


for name, model in models:
    start_time = time.time()
    model.fit(trainX, trainY)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(name)
    print('Score for evaluation set :', model.score(evalX, evalY))
