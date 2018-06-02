import matplotlib.pyplot as plt

import p2funcs as p2f
import auxfuncs as aux

from sklearn.datasets import make_circles

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel, cosine_similarity, sigmoid_kernel

imgpath = '../../figs/out/kerneldirect/'
jspath = '%sjs_plotting/' % imgpath

X, y = make_circles(n_samples=1000, factor=.3, noise=.05, random_state=12)

gamma = 0.5

x_kerns = [("linear", linear_kernel(X)), ("gaussian", rbf_kernel(X, gamma=gamma)), ("laplacian", laplacian_kernel(X, gamma=gamma)), ("cosine", cosine_similarity(X)), ("sigmoid", sigmoid_kernel(X))]



p2f.plot_scatter(X, y, '', gamma=gamma, x_label='x coordinate', y_label='y coordinate', cat1='Category 1', cat0='Category 0', output='save', path='%sinit_scatter.png' % imgpath, jspath='%sinit_scatter.js' % jspath)

#aux.simplot(X, y)

for k_lab, k_data in x_kerns:
    #aux.simplot(k_data, y)
    p2f.plot_scatter(k_data, y, '', gamma=gamma, x_label='x coordinate', y_label='y coordinate', cat1='Category 1', cat0='Category 0', output='save', path='%s%s_scatter.png' % (imgpath,k_lab), jspath='%s%s_scatter.js' % (jspath,k_lab))