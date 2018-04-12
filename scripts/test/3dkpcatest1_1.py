"""
KPCA using 3D dataset based off Sebastian Raschka swiss roll tutorial
"""

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

def stepwise_kpca(X, gamma, n_components):
    '''
    Implementation of a RBF kernel PCA.

    Arguments:
        X: An M*N dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    '''
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

#use a random 100 subset file
inp_csv = pd.read_csv('../../data/sample/rand100subset1.csv', delimiter=',', header=0)

#chosen dimensions - hard coded for now, easy to soft code later.
# blood glucose
glucose = inp_csv['glucose']
# sex (given as 1 or 2)
sex = inp_csv['sex']
# BMI
bmi = inp_csv['bmi']
# weight
weight = inp_csv['weight']

col_by_sex = []

for i in range(len(sex)):
    if sex[i] == 1:
        col_by_sex.append('red')
    if sex[i] == 2:
        col_by_sex.append('blue')

#print(col_by_sex)

#create vector
data = []

for i in range(len(bmi)):

    #create sample vector
    xi = []
    xi.append(glucose[i])
    xi.append(weight[i])
    xi.append(bmi[i])

    #Append sample vector to array
    data.append(xi)

#simple way to convert to numpy array
X = np.array(data)

#print(X)

#plot initial data
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=col_by_sex)

#ax.legend()

plt.title('Glucose vs weight vs bmi by sex')
ax.set_xlabel('Blood glucose concentration')
ax.set_ylabel('weight ')
ax.set_zlabel('BMI')
plt.show()

#2-component linear PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_spca[:, 0], X_spca[:, 1], c=col_by_sex)

plt.title('First 2 principal components after Linear PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
