import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel, cosine_similarity, sigmoid_kernel
from sklearn.decomposition import KernelPCA

import p2funcs as p2f

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

#kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
#kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
#kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
#kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

for kernel, abbreviation, kpca in kpcas:
    
    if kernel == 'Laplacian KPCA':
        X_kpca = kpca.fit_transform(K_lap)
    else:
        X_kpca = kpca.fit_transform(X)
    
    clf = SVC(kernel='linear')
    clf.fit(X_kpca, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    
    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    
    filename = '../../int_charts/vect1.txt'
    p2f.plot_write('vect X \n', filename)
    p2f.plot_write(str(xx.tolist()), filename)
    p2f.plot_write('\nvect y\n', filename)
    p2f.plot_write(str(yy.tolist()), filename)
    p2f.plot_write('\nupper y\n', filename)
    p2f.plot_write(str(yy_up.tolist()), filename)
    p2f.plot_write('\nlower y\n', filename)
    p2f.plot_write(str(yy_down.tolist()), filename)
    p2f.plot_write('\ncat0 X\n', filename)
    p2f.plot_write(str(X_kpca[y==0, 0].tolist()), filename)
    p2f.plot_write('\ncat0 Y\n', filename)
    p2f.plot_write(str(X_kpca[y==0, 1].tolist()), filename)
    p2f.plot_write('\ncat1 X\n', filename)
    p2f.plot_write(str(X_kpca[y==1, 0].tolist()), filename)
    p2f.plot_write('\ncat1 Y\n', filename)
    p2f.plot_write(str(X_kpca[y==1, 1].tolist()), filename)
    
    
    
    

    
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
    
    
    plt.title('3-Class classification using Support Vector Machine with custom'
              ' kernel')
    plt.axis('tight')
    plt.show()
    plt.close
    
#p2f.plotwrite()
