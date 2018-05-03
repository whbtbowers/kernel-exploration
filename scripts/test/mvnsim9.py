#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:57:57 2018

@author: whb17
"""
import seaborn as sns
sns.set(style="ticks")
sns.set(style='darkgrid')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from numpy.random import multivariate_normal
'''
# To label simulation
simname = input('Input simulation name: ')
simname = str(simname)

# Name of script to trace where images came from
scriptname = 'mvnsim 9'

filepath = '../../figs/out/%s/%s/' % (scriptname, simname)

if not os.path.exists(filepath):
    os.makedirs(filepath)
'''
#To show runtime of script
StartTime = time.time()

#Set up target array
target = np.zeros(100, dtype=int)
target[0:50] = '1'


# Generate covariance matrix
mat = np.zeros([100, 500], dtype=int)
mat.reshape((100, 500))
mat_diag
print(mat.shape)

#diag_mat = np.diag(mat)
#print(diag_mat)
# Mean and array size for non-discriminatory variables
nd_size = [400, 100]
nd_mean = [40, 10]
#cov = [[1, 0], [0, 100]]

nd_x, nd_y = multivariate_normal(nd_mean, cov, nd_size).T

#metrics for discriminatory variables
d_size = [100, 50]
d1_mean = [60, 15]
d2_mean = [20, 5]



d1_x, d1_y = multivariate_normal(d1_mean, cov, d_size).T
#d1_X = multivariate_normal(d1_mean, cov, d_size).T

d2_x, d2_y = multivariate_normal(d2_mean, cov, d_size).T
#d2_X = multivariate_normal(d2_mean, cov, d_size).T

#combine two half-columns of discriminatory variables into full columns
dvars = np.vstack((d1_x, d2_x))
#print(dvars.shape)

#combine discriminatory varibales into array with non-discriminatory variables
mvn_sim = np.hstack((dvars, nd_x))
#mvn_sim = np.hstack((mvn_sim, target))
#df_notarget = pd.DataFrame.from_records(mvn_sim)
#print(mvn_sim.shape)

#convert array to dataframe    
mvn_sim_df = pd.DataFrame.from_records(mvn_sim)

#print(mvn_sim_df[0])
#Create boxplot
#mvn_sim_df.boxplot(by=mvn_sim_df[500])

#print(mvn_sim_df)
      

plt.figure(figsize=(50,15))

plt.subplot(2, 1, 1)
img1 = sns.boxplot(data=mvn_sim_df[target==1])
plt.title("Category A distribution of dataset %s" % simname, fontsize=20)

plt.subplot(2, 1, 2)
img2 = sns.boxplot(data=mvn_sim_df[target==0])
plt.title("Category B distribution of dataset %s" % simname, fontsize=20)

plt.savefig('../../data/simulated/mvnsim/mvnsim%sdist.png' % simname)
#plt.show()

plt.close()

mvn_sim_df = mvn_sim_df
print(mvn_sim_df.shape)

mvnsim_csv_path = '../../data/simulated/mvnsim/mvnsim%s.csv' % simname
mvn_sim_df.to_csv(path_or_buf=mvnsim_csv_path, sep=',')

#Write target data to .npy file
np.save('../../data/simulated/mvnsim/target' + simname + '.npy', target)


print('Dataset %s saved' % simname)

#Calculate and display time taken or script to run 
EndTime = (time.time() - StartTime)
print('Time taken for script to run is %.2f seconds' % EndTime)

