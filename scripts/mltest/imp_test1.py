import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

from p2funcs import plot_scatter

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import scale, normalize, Imputer
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy import interp


#To show runtime of script
StartTime = time.time()

# Name of script to trace where images came from
scriptname = 'imp_test_1.py'

#Select current toy dataset
#dataset = '013'
dataset = 'mesa2'

#Create directory if directory does not exist
filepath = '../../figs/out/%s/%s/' % (scriptname, dataset)

if not os.path.exists(filepath):
    os.makedirs(filepath)

#Import toy data and target
#X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
X = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

#print(X)


# Make list of sample IDs
row_names = X.index.values.tolist()

# Binarise 'Yes' and 'No'
X = X.replace('Yes', 1)
X = X.replace('No', 0)

# Display metrics for initial data
n_cols, n_rows = X.shape
print('\nInitial data contains %s columns and %s rows.' % (n_cols, n_rows))
col_names = list(X)
#print('Categories:')
#print(col_names)

# Remove ID column and display metrics
X = X.drop(columns='study')

n_cols, n_rows = X.shape
print("\nAfter dropping 'study' column, data contains %s columns and %s rows." % (n_cols, n_rows))

col_names = list(X)

# Initiate counter to count number of null values per column
dropped_cols = 0

# Find columns containing <10% of filled fields
for i in col_names:
    
    # Initiate counter to count number of null values per column
    null_cells = 0
    
    for j in X[i]:
        if pd.isnull(j) == True:
            null_cells += 1

    # Remove column if more than 10% values empty
    if null_cells / len(X[i]) >= 0.9:
        X = X.drop(columns=i)
        dropped_cols += 1

n_cols, n_rows = X.shape        
print('\n%s columns in dataset found to have <10%% of cells populated.' % dropped_cols)
print('\nAfter columns <= 10%% populated removed, data contains %s columns and %s rows.' % (n_cols, n_rows))

col_names = list(X)

#Find rows containing <10% of filled fields
for i in row_names:

#    print(X.loc[i])
    # Initiate counter to count number of null values per row
    nulls = 0

    for j in X.loc[i]:
        if pd.isnull(j) == True:
            nulls += 1
    # Remove row if more than 10% values empty
    if nulls/len(X.loc[i]) >= 0.9:
        print(i)
        X = X.drop(index=i)

n_cols, n_rows = X.shape
print('\nAfter columns and rows <= 10 percent populated removed, data contains %s columns and %s rows.' % (n_cols, n_rows))


# Convert dataframe to numpy matrix for scikit learn
X = X.as_matrix()

# Uses mean as imputation strategy
impute = Imputer()
X_imputed = impute.fit_transform(X)

#print(X_imputed.shape)
n_cols, n_rows = X.shape
print('\nAfter imputation, data contains %s columns and %s rows.' % (n_cols, n_rows))

#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print('\nTime taken for script to run is %.2f seconds.' % EndTime)
