import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA

#Generates two interlocked crescents, each with 25 points (ie n_samples = 50, 50 points overall.)
#random_state=1 is RandomState instance generated by np.random. random dataset generated by seed of 1.
## DO NOT COMMENT OUT, DATASET USED IN ALL EXAMPLES ##
X, y = make_moons(n_samples=50, random_state=1)

# Red half moon
plt.scatter(X[y==0, 0], X[y==0, 1], # Start and peak/trough of each 'moon'.
            color ='red', marker='o', alpha=0.5)

#Blue half moon
plt.scatter(X[y==1, 0], X[y==1, 1], # Start and peak/trough of each 'moon'.
            color='blue', marker='^', alpha=0.5)

plt.xlabel('x coordinate')
plt.ylabel('y coordinate')

#plt.show()
plt.savefig('../figs/tutorial/mlxtendex1_1.png')
plt.close()
# Moons are linearly inseperable so standard linear PCA will fail to accurately represent data in 1D space.

#Use PCA for dimensionality reduction

#specify number of components in PCA
pca = PCA(n_components=2)
#Transform X in accordance with 2-component PCA
X_pca = pca.fit(X).transform(X)

# Red half moon
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], # Start and peak/troughof each 'moon'.
            color ='red', marker='o', alpha=0.5)

#Blue half moon
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], # Start and peak/troughof each 'moon'.
            color='blue', marker='^', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')

#plt.show()
plt.savefig('../figs/tutorial/mlxtendex1_2.png')
plt.close()

#This shows linear PCA unable to generate subspace suitable to linearly separate data.
#PCA is unsupevised method, so input data unlabeled.

# Radial base function (RBF) kernel PCA (KPCA)
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import RBFKernelPCA as KPCA

#Specify 2-component PCA, gamma choice dependent on dataset, obtained via hyperparameter methods such as Grid search.
#Gamma here is gamma said to give 'good' results by creator of tutorial.
kpca = KPCA(gamma=15.0, n_components=2)
#Fit X with above KPA specifications
kpca.fit(X)
#Project X values onto 'new' (higher dimensional?) feature space (rep by 'g' in associated notes)
X_kpca = kpca.X_projected_

#Plot moons but with kernel-projected X
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
            color='red', marker='o', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='^', alpha=0.5)

plt.title('First 2 principal components after RBF Kernel PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex1_3.png')
plt.close()

#New feature space now linearly seperable at x=0. This would be fine for SVM, but PCA has focuson dimensionality reduction.

import numpy as np

plt.scatter(X_kpca[y==0, 0], np.zeros((25, 1)), #Uses matrix of zeroes to reduce to single dimension
            color='red', marker='o', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], np.zeros((25, 1)), #Uses matrix of zeroes to reduce to single dimension
            color='blue', marker='^', alpha=0.5)

plt.title('First principal component after RBF Kernel PCA')
plt.xlabel('PC1')
plt.yticks([])
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex1_4.png')
plt.close()
 # Feature space linearly seperable at x=0, also data entirely 1-D, forming horizontal line.

 #subspace like this can then be used as input in generalised classification models eg logistic regression.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

#New data, two crescents, now using 100 samples per crescent,using random state seed of 5.
X2, y2 = make_moons(n_samples=200, random_state=5)
#Transform new dataset according to previous KPCA parameters.
X2_kpca = kpca.transform(X2)

#Initial data
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
            color='red', marker='o', alpha=0.5, label='fit data')
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='^', alpha=0.5, label='fit data')

#New data projected onto new component axes
plt.scatter(X2_kpca[y2==0, 0], X2_kpca[y2==0, 1],
            color='orange', marker='v',
            alpha=0.2, label='new data')
plt.scatter(X2_kpca[y2==1, 0], X2_kpca[y2==1, 1],
            color='cyan', marker='s',
            alpha=0.2, label='new data')

plt.legend()
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex1_5.png')
plt.close()
