#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:00:02 2018

@author: whb17
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import make_classification

#label dataset
simname = input('Input simulation name: ')
simname = str(simname)

#generate multivariate dataset and test values
X,y = make_classification (n_samples=100,
                              n_features=500,
                              n_informative=100,
                              n_redundant=400,
                              n_classes=2,
                              class_sep=5.0,
                              )

plt.figure(figsize=(8, 6))

cata = plt.scatter(X[y==0, 0],
                   X[y==0, 1],
                   color='red',
                   alpha=0.5
                   )

catb = plt.scatter(X[y==1, 0],
                   X[y==1, 1],
                   color='blue',
                   alpha=0.5)

plt.title('Initial data')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.legend([cata, catb],['Category A', 'Category B'])

plt.show()
#plt.savefig('../../figs/out/%s/%s/initial.png' % (scriptname, dataset))
plt.close()

# Convert X to dataframe
X_df = pd.DataFrame.from_records(X)

#Write X to csv
X_csv_path = '../../data/simulated/mvnsim/mvnsim'+ simname + '.csv'
X_df.to_csv(path_or_buf=X_csv_path, sep=',', line_terminator='\n')

#Write y to .npy file
np.save('../../data/simulated/mvnsim/target' + simname + '.npy', y)

print('Dataset %s saved successfully.' % simname)