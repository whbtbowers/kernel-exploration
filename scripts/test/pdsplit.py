#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:20:33 2018

@author: whb17
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Function for splitting target column from pandas dataset
def target_split(df, col):
    
    df_mat = df.as_matrix()
    df_mat = df_mat.T
    rows, cols = df_mat.shape    
    target = df_mat[:, [col]]
    target = target.reshape(1, len(target))    
    data = df_mat[:, 0:cols-1]
    
    return(data.T, target[0])

dataset = '014'

# Import toy dataset
inp_csv = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)

X, y = target_split(inp_csv, 500)

print(X.shape)
print(y)