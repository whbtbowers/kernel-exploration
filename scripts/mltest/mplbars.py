import matplotlib.pyplot as plt
import numpy as np
from pylab import get_cmap
import matplotlib.colors as mcolors

def mpl_simplebar(x, y, xlab, ylab, col_list):
    
    fig, ax = plt.subplots()
    plt.bar(x, y, color=col_list)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.show()
    
def get_col_list(colormap, n_cols):
    
    colours = []
    
    cmap = get_cmap(colormap, n_cols)
    
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 to get rgb
        colours.append(mcolors.rgb2hex(rgb))
    
    return(colours)
    
x = ['Bill', 'Fred', 'Mary', 'Sue']
money = [1.5, 2.5, 5.5, 2.0]

mpl_simplebar(x, money, 'Outcome', 'Mean AUC after 10-fold cross-validation', get_col_list('autumn', len(x)))
