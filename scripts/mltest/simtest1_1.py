#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:30:17 2018

@author: whb17

First attempt at preprocessing, KNN, and CV
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import preprocessing as prep
from sklearn.cross_decomposition import PLSRegression

# Read in simulated data and target categories

# preloaded diabetes data
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

#X = pd.read_csv('../../data/simulated/mvnsim/mvnsim003.csv', delimiter=',', header=0)
#y = pd.read_csv('../../data/simulated/mvnsim/target003.csv', delimiter=',', header=0)

#print(X._csv.shape)

#X = X_csv.as_matrix
#y = y_csv.as_matrix

## PREPROCESSING ##

X_scaled = prep.scale(X)

## PLS ##

