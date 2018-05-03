"""
KPCA using method in Sebastian Rascka tutorial
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
plt.ion()


def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: An M*N dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

#import subset of initial csv
inp_csv = pd.read_csv('../../data/sample/rand100subset1.csv', delimiter=',', header=0)

#plot based on vector
#input vector (eg glucose & bmi)
#category (eg sex)
#out_type - output type, 'show' or 'save'

#chosen dimensions
glucose = inp_csv['glucose']
sex = inp_csv['sex']
bmi = inp_csv['bmi']


#Create empty array to add vectors to
data = []

for i in range(len(bmi)):

    #create sample vector
    xi = []
    xi.append(bmi[i])
    xi.append(glucose[i])

    #Append sample vector to array
    data.append(xi)

#simple way to convert to numpy array
X = np.array(data)
print(np.cov(X))

#set initial gamma
gamma = 100

#set interval of gamma increase
interval = 1.0

gamma_max = 200
'''
while gamma <= gamma_max:

    print('gamma: ' + str(gamma))

    X_kpca = stepwise_kpca(X, gamma=gamma, n_components=2)

    #Graph after kpca
    #generate graph from matrix
    for i in range(len(X)):
        if sex[i] == 1:
            #Sex 1 glucose v BMI
            sex1 = plt.scatter(X_kpca[i][0], #bmi
                X_kpca[i][1],    #glucose
                color ='red',
                marker='o',     #circle marker
                alpha=0.5,
                )
        elif sex[i] == 2:
            sex2 = plt.scatter(X_kpca[i][0], #bmi
                X_kpca[i][1],    #glucose
                color='blue',
                marker='^',     #triangle marker
                alpha=0.5,
                )

    plt.title('BMI vs glucose by sex after kernel PCA (gamma=' + str(gamma) + ')')
    plt.ylabel('Serum glucose concentration')
    plt.xlabel('BMI')
    plt.legend([sex1, sex2], ['Sex 1', 'Sex 2'])

    plt.show()
    plt.pause(0.005)
    plt.clf()
    #plt.savefig('../../figs/bivariate/subsetkpca1_4_gamma' + str(gamma) + '.png')

    gamma += interval

plt.show()
'''
