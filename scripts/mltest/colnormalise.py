#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:28:29 2018

@author: whb17
"""

import pandas as pd
import numpy as np

in_csv = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_calcs.csv', sep=',', header=0, index_col=1)

out_csv = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

num = []

x = 5

#print(isinstance(x, int))

for i in range(len(in_csv["icam1"])):
    print(pd.Series(in_csv["icam1"][i]))
    #if isinstance(in_csv["icam1"][i], int):
    #if type(in_csv["icam1"][i]) == int:
        #print(i)
