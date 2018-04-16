#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:41:19 2018

@author: whb17
"""

import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import pandas as pd
'''
# To label simulation
simname = input('Input simulation name: ')
simname = str(simname)
'''
# Initiate array for target data

target = [['0'*50 + '1'*50]]

# generating non-discriminatory variables

nd_mean = [60, 50]
nd_cov = [[1,0], [0,100]]
nd_num_rows = 100
nd_num_cols = 400

nd_mvn = rnd.multivariate_normal(nd_mean, nd_cov, (nd_num_rows, nd_num_cols))

print(nd_mvn)

# generating first set of discriminatory variables

dmean1 = [30, 70]
dcov1 = [[1,0], [0,100]]
dnum_rows1 = 50
dnum_cols1 = 100

dmvn1 = rnd.multivariate_normal(dmean1, dcov1, (dnum_rows1, dnum_cols1))


# generating second set of discriminatory variables

dmean2 = [90, 40]
dcov2 = [[1,0], [0,100]]
dnum_rows2 = 50
dnum_cols2 = 100

dmvn2 = rnd.multivariate_normal(dmean2, dcov2, (dnum_rows2, dnum_cols2))

    
#combine two half-columns of discriminatory variables into full columns
dvars = np.vstack((dmvn1, dmvn2))

#combine discriminatory varibales into array with non-discriminatory variables
mvn_sim = np.hstack((dvars, nd_mvn))

#create arbitrary CSV headers
heads = []

for i in range(500):
    heads.append('var_' + str(i+1))

#convert array to dataframe    
mvn_sim_df = pd.DataFrame.from_records(mvn_sim)
print(mvn_sim_df)

'''
#convert dataframe to CSV file to be used by other scripts
mvnsim_csv_path = '../../data/simulated/mvnsim/mvnsim' + simname + '.csv'
mvnsimdf.to_csv(path_or_buf=mvnsim_csv_path, sep=',', header=heads)

#create CSV for target values
target = np.array(target)
targetdf = pd.DataFrame.from_records(target)
target_csv_path = '../../data/simulated/mvnsim/target' + simname + '.csv'
targetdf.to_csv(path_or_buf=target_csv_path, sep=',', header='target')
'''