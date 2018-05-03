#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:35:31 2018

@author: whb17
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
sns.set(style="ticks")
sns.set(style='whitegrid')

import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')


def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', output=None, path=None, ply_title=None):
    
    fig = plt.figure(figsize=(8, 6))
    
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
    plt.ylabel(y_label)    
    gamma_label = mpatches.Patch(color='white', label='gamma')
    plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category A', 'Category B'])
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    elif output == 'plotly':        
        plotly_fig = tls.mpl_to_plotly(fig)
        plot_url = py.plot(plotly_fig, filename=ply_title)
    else:
        pass
        
    plt.close() 

def target_split(df, col):
    
    df_mat = df.as_matrix()
    #df_mat = df_mat.T
    rows, cols = df_mat.shape    
    target = df_mat[:, [col]]
    target = target.reshape(1, len(target))    
    data = df_mat[:, 0:cols-1]
    
    return(data.T, target[0])

def distribution_boxplot(df, targ, a_title, b_title, output=None, path=None, ply_title=None):
    
    fig = plt.figure(figsize=(50,15))
    
    plt.subplot(2, 1, 1)
    img1 = sns.boxplot(data=df[targ==1])
    plt.title(a_title, fontsize=20)
    
    plt.subplot(2, 1, 2)
    img2 = sns.boxplot(data=df[targ==0])
    plt.title(b_title, fontsize=20)
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    elif output == 'plotly':
        plotly_fig = tls.mpl_to_plotly(fig)
        plot_url = py.plot(plotly_fig, filename=ply_title)
    else:
        pass
        
    plt.close() 