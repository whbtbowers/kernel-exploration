import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel, cosine_similarity, sigmoid_kernel
from sklearn.decomposition import KernelPCA

X, y = make_circles(n_samples=1000, factor=.3, noise=.05, random_state=12)

cata = plt.scatter(X[y==0, 0],
                   X[y==0, 1],
                   color='red',
                   marker = '^',
                   alpha=0.5
                   )

catb = plt.scatter(X[y==1, 0],
                   X[y==1, 1],
                   color='blue',
                   marker = 's',
                   alpha=0.5
                   )
plt.show()
plt.close()

gamma = 5

K_lap = laplacian_kernel(X, gamma=gamma)

kpcas = []

kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

for kernel, abbreviation, kpca in kpcas:
    
    if kernel == 'Laplacian KPCA':
        X_kpca = kpca.fit_transform(K_lap)
    else:
        X_kpca = kpca.fit_transform(X)

    h = .02  # step size in the mesh


    # we create an instance of SVM and fit out data.
    clf = SVC(kernel='linear', gamma=gamma)
    clf.fit(X_kpca, y)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_kpca[:, 0].min() - 1, X_kpca[:, 0].max() + 1
    y_min, y_max = X_kpca[:, 1].min() - 1, X_kpca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c='y', cmap=plt.cm.Paired, edgecolors='k')
    
    cata = plt.scatter(X_kpca[y==0, 0],
                       X_kpca[y==0, 1],
                       color='red',
                       cmap=plt.cm.Paired,
                       marker = '^',
                       alpha=0.5
                       )
    
    catb = plt.scatter(X_kpca[y==1, 0],
                       X_kpca[y==1, 1],
                       color='blue',
                       marker = 's',
                       alpha=0.5
                       )
    
    
    plt.title('3-Class classification using Support Vector Machine with custom'
              ' kernel')
    plt.axis('tight')
    plt.show()
    plt.close

'''
gamma=5

K_lap = laplacian_kernel(X, gamma=gamma)

kpcas = []

kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

for kernel, abbreviation, kpca in kpcas:
    
    if kernel == 'Laplacian KPCA':
        X_kpca = kpca.fit_transform(K_lap)
    else:
        X_kpca = kpca.fit_transform(X)
        
    cata = plt.scatter(X_kpca[y==0, 0],
                       X_kpca[y==0, 1],
                       color='red',
                       marker = '^',
                       alpha=0.5
                       )

    catb = plt.scatter(X_kpca[y==1, 0],
                       X_kpca[y==1, 1],
                       color='blue',
                       marker = 's',
                       alpha=0.5
                       )
    plt.show()
    plt.close()
'''