"""
This script is for generating a multivariate normal simulation dataset.
"""

import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import pandas as pd

# User input graphnames
simname = input('Input simulation name: ')
simname = str(simname)

# Initiate array for target data

target = ['0'*50 + '1'*50]

# generating non-discriminatory variables

nd_mean = [60, 50]
nd_cov = [[1,0], [0,100]]
nd_num_rows = 100
nd_num_cols = 400

nd_mvn = rnd.multivariate_normal(nd_mean, nd_cov, (nd_num_rows, nd_num_cols))
ndx1, ndy1 = mvn1.T



# generating second set of discriminatory variables

dmean1 = [30, 70]
dcov1 = [[1,0], [0,100]]
dnum_rows1 = 100
dnum_cols1 = 50

dmvn1 = rnd.multivariate_normal(dmean1, dcov1, (dnum_rows1, dnum_cols1))
dx1, dy1 = mvn2.T

# generating second set of discriminatory variables

dmean2 = [90, 40]
dcov2 = [[1,0], [0,100]]
dnum_rows2 = 100
dnum_cols2 = 50

dmvn2 = rnd.multivariate_normal(dmean2, dcov2, (dnum_rows2, dnum_cols2))
dx2, dy2 = mvn2.T

#join 3 datasets into 1
mvnsim = np.hstack((nd_mvn, dmvn1))
mvnsim = np.hstack((mvnsim, dmvn2))
x, y = mvnsim.T

# convert array to dataframe
mvnsimdf = pd.DataFrame.from_records(mvnsim)

#generate stand-in variable headers
heads = []

for i in range(500):
    heads.append('var_' + str(i+1))


#convert dataframe to CSV file to be used by other scripts
mvnsim_csv_path = '../../data/simulated/mvnsim/2mvnsim' + simname + '.csv'
#mvnsimdf.to_csv(path_or_buf=mvnsim_csv_path, sep=',', header=heads)
'''
#create CSV for target values
target = np.array(target)
targetdf = pd.DataFrame.from_records(target)
target_csv_path = '../../data/simulated/mvnsim/target' + simname + '.csv'
targetdf.to_csv(path_or_buf=target_csv_path, sep=',', header='target')
'''
'''
group1 = plt.scatter(x1,
    y1,
    color ='magenta',
    marker='s',     #square marker
    alpha=0.5,
    )

group2 = plt.scatter(x2,
    y2,
    color='cyan',
    marker='p',     #pentagon marker
    alpha=0.5,
    )

plt.title('Random multivariate normal simulated dataset ' + simname)
plt.legend([group1, group2], ['Group 1', 'Group 2'])

plt.show()
#plt.savefig('../../figs/mvnfigs/mvngroups' + simname + '.png')
plt.close()
'''
sim = plt.scatter(x,
    y,
    color ='green',
    marker='o',     #square marker
    alpha=0.5,
    )

plt.title('Random multivariate normal simulated dataset ' + simname)

plt.show()
#plt.savefig('../../figs/mvnfigs/mvnmono' + simname + '.png')
plt.close()