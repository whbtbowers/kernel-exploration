#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:04:06 2018

@author: whb17
"""

import numpy as np

mat = np.array([[1,2],[3,4],[4,4]])

print(np.amax(mat))

for i in range(len(mat)):
    for j in range(len(mat[i])):
        if mat[i][j] == np.amax(mat):
            print((i,j))