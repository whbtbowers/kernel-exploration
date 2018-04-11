from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# 1000 samples, slightly noisy, ring between +/- 0.5 surrounded by ring between +/- 1.0
X, y = make_circles(n_samples=1000, random_state=123,
                    noise=0.1, factor=0.2)

plt.figure(figsize=(8,6))


#alpha in case of pyplot is opacity.
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)
plt.title('Concentric circles')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex2_1.png')
plt.close()


from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import RBFKernelPCA as KPCA

#2-component RBF KPCA
kpca = KPCA(gamma=15.0, n_components=2)

#X fit to specified KPCA parameters
kpca.fit(X)

# Fit X projected to new feature space.
X_kpca = kpca.X_projected_

plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
            color='red', marker='o', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='^', alpha=0.5)

plt.title('First 2 principal components after RBF Kernel PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex2_2.png')
plt.close()

#Results large, blue, disperse reion and smaller, red, dense region, now linearly seperable.

import numpy as np

# Reduce dimensionality to 1
plt.scatter(X_kpca[y==0, 0], np.zeros((500, 1)), #Dimensionality reduced to 1 by matrix of zeroes
            color='red', marker='o', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], np.zeros((500, 1)),
            color='blue', marker='^', alpha=0.5)

plt.title('First principal component after RBF Kernel PCA')
plt.xlabel('PC1')
plt.yticks([])
#plt.show()
plt.savefig('../figs/tutorial/mlxtendex2_3.png')
plt.close()
