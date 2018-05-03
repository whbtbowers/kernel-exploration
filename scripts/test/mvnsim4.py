
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:06:18 2018
@author: whtbowers
"""

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#To label output files

simname = input('Input simulation name: ')
simname = str(simname)

#Generate non-discriminarory variables

nd_mvn = []

for i in range(100):
        
    row = []
    
    for j in range(400):
        row.append(multivariate_normal.rvs(mean=50, cov=10))
    
    nd_mvn.append(row)

nd_mvnm = np.array(nd_mvn)

#Generate first set of discriminatory variables

dmvn1 = []

for i in range(50):
        
    row = []
    
    for j in range(100):
        row.append(multivariate_normal.rvs(mean=30, cov=10))
    
    dmvn1.append(row)

dmvn1 = np.array(dmvn1)

#Generate second set of discriminatory variables

dmvn2 = []

for i in range(50):
        
    row = []
    
    for j in range(100):
        row.append(multivariate_normal.rvs(mean=70, cov=10))
    
    dmvn2.append(row)

dmvn2 = np.array(dmvn2)

# vertically join the discriminatory dataframes

dmvn = np.vstack((dmvn1, dmvn2))
#print(dmvn.shape)

# Add discriminatory columns to non-discriminatory values dataframe

mvnsim = np.hstack((nd_mvn, dmvn))
mvnsim_df = pd.DataFrame.from_records(mvnsim)
#print(mvnsim_df)

#create arbitrary CSV headers
heads = []

for i in range(500):
    heads.append('var_' + str(i+1))

#create target values for classification
target = []
    
for i in range(50):
    target.append('0')
    
for i in range(50):
    target.append('1')
tagethead = ['target']
target = np.array(target)
targetdf = pd.DataFrame.from_records(target)
target_csv_path = '../../data/simulated/mvnsim/target' + simname + '.csv'
targetdf.to_csv(path_or_buf=target_csv_path, sep=',', header='targethead')

#convert dataframe to CSV file to be used by other scripts
mvnsim_csv_path = '../../data/simulated/mvnsim/mvnsim' + simname + '.csv'
mvnsim_df.to_csv(path_or_buf=mvnsim_csv_path, sep=',', header=heads)
