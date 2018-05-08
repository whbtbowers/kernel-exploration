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

# To label simulation
simname = input('Input simulation name: ')
simname = str(simname)

# Name of script to trace where images came from
scriptname = 'mvnsim 9'

filepath = '../../figs/out/%s/%s/' % (scriptname, simname)

if not os.path.exists(filepath):
    os.makedirs(filepath)

#To show runtime of script
StartTime = time.time()

#Set up target array
target = np.zeros(100, dtype=int)
target[0:50] = '1'


#metrics for discriminatory variables
d_size = [500, 50]

#1st component
d1_mean = [20.5, 15.5]
cov_1 = [[9, 6], [6, 6]]
d1_x, d1_y = multivariate_normal(d1_mean, cov_1, d_size).T

#2nd component
d2_mean = [20, 15]
cov_2 = [[12, 8],[8, 8]]
d2_x, d2_y = multivariate_normal(d2_mean, cov_2, d_size).T


# Combine to create full 100*500 dataset
mvn_sim = np.vstack((d1_x, d2_x))

mvn_sim_df = pd.DataFrame.from_records(mvn_sim)



plt.figure(figsize=(50,15))

plt.subplot(2, 1, 1)
img1 = sns.boxplot(data=mvn_sim_df[target==1])
plt.title("Category 1 distribution of dataset %s" % simname, fontsize=20)

plt.subplot(2, 1, 2)
img2 = sns.boxplot(data=mvn_sim_df[target==0])
plt.title("Category 0 distribution of dataset %s" % simname, fontsize=20)

plt.savefig('../../data/simulated/mvnsim/mvnsim%sdist.png' % simname)
#plt.show()
plt.close()


mvnsim_csv_path = '../../data/simulated/mvnsim/mvnsim%s.csv' % simname
mvn_sim_df.to_csv(path_or_buf=mvnsim_csv_path, sep=',')

# Write target data to .npy file
np.save('../../data/simulated/mvnsim/target' + simname + '.npy', target)


print('\nDataset %s saved' % simname)

#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print('\nTime taken for script to run is %.2f seconds' % EndTime)
