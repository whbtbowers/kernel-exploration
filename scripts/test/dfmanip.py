#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:52:33 2018

@author: whb17
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

#print(df)

print(pd.Series.unique(df['strkisch']))
#X = pd.DataFrame(X.ses.str.split(':',1).tolist(), columns = ['ses', 'ses_labels'])