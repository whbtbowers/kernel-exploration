#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:30:10 2018

@author: whb17
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles

#label dataset
simname = input('Input simulation name: ')
simname = str(simname)

X,y = make_circles(n_samples=500,
                   noise = 0.1,
                   factor = 0.5,
                   random_state=123,                                              
                   )

#Plot data
plt.figure(figsize=(10, 7))

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
#np.save('../../data/simulated/mvnsim/target' + simname + '.npy', y)

#print('Dataset %s saved successfully.' % simname)