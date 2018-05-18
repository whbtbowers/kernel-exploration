import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import p2funcs as p2f

inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',')        
inp_df = p2f.filt_imp(inp_df, 0.1)


def toybox_gen(inp_df):
    
    
    #Define size of component using input data
    cols, rows = inp_df.shape
    comp_size = [cols, int(round(rows/2))]
    
    cov1 = np.array([[[0.3, 0.2], [0.2, 0.2]], [[0.6, 0.4], [0.4, 0.4]], [[1.2, 0.8], [0.8, 0.8]], [[2.4, 1.6], [1.6, 1.6]], [[6, 4], [4, 4]],[[9, 6], [6, 6]]])
    cov2 = np.array([[12, 8],[8, 8]])
    
    mean1 = np.array([[20, 15], [20, 15], [20, 15], [20, 15], [20, 15], [20.5, 15.5]])
    mean2 = [20, 15]
    
    #Set up target array
    target = np.zeros(100, dtype=int)
    target[0:50] = '1'
    
    dataset_list = []
    
    # counter for labelling dataset
    counter = 1
    # Second component consistent. Generate first to save time.
    d2_x, d2_y = multivariate_normal(mean2, cov2, comp_size).T
    
    for i in range(len(cov1)):
        
        d1_x, d1_y = multivariate_normal(mean1[i], cov1[i], comp_size).T
        
        #Put components together
        mvn_sim = np.vstack((d1_x, d2_x))
        mvn_sim_df = pd.DataFrame.from_records(mvn_sim)
        dataset_list.append(('ds00%d' % counter, mvn_sim_df))
        counter += 1
        
    return(dataset_list, target)

ds_list, y = toybox_gen(inp_df)

for label, dataset in ds_list:
    print(label)    