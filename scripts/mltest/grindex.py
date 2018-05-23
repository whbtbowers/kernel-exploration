#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:46:57 2018

@author: whb17
"""
import numpy as np
import p2funcs as p2f

x1 = ['a', 'b', 'c']

x2 = ['d', 'e', 'f']


#i_index, j_index = np.where(mat == chos_x )


mat1 = np.array([[1, 2, 3], [4, 5, 6], [4, 8, 9]])
mat2 = np.array([[11, 21, 31], [41, 51, 61], [71, 81, 91]])
mat3 = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]])

matdict = {
        '1st':mat1,
        '2nd':mat2,
        '3rd':mat3
        }

choicedict = '3rd'

#print(matdict[choicedict])

dictmat = matdict[choicedict]

max_auc = np.max(dictmat)

i_index, j_index = np.where(dictmat == max_auc)

print('Corresponding element in x1: %s\nCorresponding element in x2: %s' % (x1[i_index[0]], x2[j_index[0]]))

array1 = [2, 2, 3, 3, 3, 4, 3, 5, 3, 6, 8]

print(p2f.most_common(array1))