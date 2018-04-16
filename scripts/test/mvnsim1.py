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

# Initiate array for arget data

target = []

# generating first group

mean1 = [10,70]
cov1 = [[1,0], [0,100]]
num_rows1 = 50
num_cols1 = 500

mvn1 = rnd.multivariate_normal(mean1, cov1, (num_rows1, num_cols1))
x1, y1 = mvn1.T # for plotting two distributions seperately

# Update target array

for i in range(num_rows1):
    target.append('0')

# generating second group

mean2 = [60, 50]
cov2 = [[1,0], [0,100]]
num_rows2 = 50
num_cols2 = 500

mvn2 = rnd.multivariate_normal(mean2, cov2, (num_rows2, num_cols2))
x2, y2 = mvn2.T# for plotting two distributions seperately

# Update target array

for i in range(num_rows2):
    target.append('1')

#join 2 datasets into 1
mvnsim = np.vstack((mvn1, mvn2))
x, y = mvnsim.T

# convert array to dataframe
mvnsimdf = pd.DataFrame.from_records(mvnsim)

#generate stand-in variable headers
heads = []

for i in range(num_cols1):
    heads.append('var_' + str(i+1))


#convert dataframe to CSV file to be used by other scripts
mvnsim_csv_path = '../../data/simulated/mvnsim/2mvnsim' + simname + '.csv'
mvnsimdf.to_csv(path_or_buf=mvnsim_csv_path, sep=',', header=heads)

#create CSV for target values
target = np.array(target)
targetdf = pd.DataFrame.from_records(target)
target_csv_path = '../../data/simulated/mvnsim/target' + simname + '.csv'
targetdf.to_csv(path_or_buf=target_csv_path, sep=',', header='target')

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

#plt.show()
plt.savefig('../../figs/mvnfigs/mvngroups' + simname + '.png')
plt.close()

sim = plt.scatter(x,
    y,
    color ='green',
    marker='o',     #square marker
    alpha=0.5,
    )

plt.title('Random multivariate normal simulated dataset ' + simname)

#plt.show()
plt.savefig('../../figs/mvnfigs/mvnmono' + simname + '.png')
plt.close()
