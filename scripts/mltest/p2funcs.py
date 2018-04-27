#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:35:31 2018

@author: whb17
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', output=None, path=None):
    
    plt.figure(figsize=(8, 6))
    
    cata = plt.scatter(x[y==0, 0],
                       x[y==0, 1],
                       color='red',
                       marker = '^',
                       alpha=0.5
                       )
    
    catb = plt.scatter(x[y==1, 0],
                       x[y==1, 1],
                       color='blue',
                       marker = 's',
                       alpha=0.5)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('y coordinate')    
    gamma_label = mpatches.Patch(color='white', label='gamma')
    plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category A', 'Category B'])
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    else:
        pass
        
    plt.close()  