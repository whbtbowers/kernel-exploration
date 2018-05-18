#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:29:55 2018

@author: whb17
"""
import pandas as pd

inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcolOC.csv', header=None, index_col=0, sep=',')


def tsplit(inp_df):

    target = inp_df[1].as_matrix()
    data =  inp_df.drop(columns=1).as_matrix()
    
    return(data, target)

X, y = tsplit(inp_df) 

print('X')
print(X)
print('y')
print(y)