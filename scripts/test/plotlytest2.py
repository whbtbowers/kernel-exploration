#import pprint as pp
import pprint
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plotly.plotly as py
import plotly.tools as tls

from sklearn.preprocessing import scale
from sklearn.decomposition import KernelPCA

print(__doc__)

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim022.csv', sep=',', header=0, index_col=0)
y = np.load('../../data/simulated/mvnsim/target022.npy')
print('Data loaded')
print(y)
X_scaled = scale(X)
X_scaled_df = pd.DataFrame.from_records(X_scaled)

gamma = 0.002

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)


X_kpca = kpca.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))

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
                   alpha=0.5)

plt.title('title')
plt.xlabel('x_label')
plt.ylabel('y_label')    
gamma_label = mpatches.Patch(color='white', label='gamma')
plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category 1', 'Category 0'])




# Converting to Plotly's Figure object..
plotly_fig = tls.mpl_to_plotly(fig)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(plotly_fig['layout'])
py.iplot(plotly_fig, filename='sine-plot')
