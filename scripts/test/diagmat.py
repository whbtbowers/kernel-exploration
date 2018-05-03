#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:18:03 2018

@author: whb17
"""
import numpy as np
import numpy.random as random
import numpy.linalg as linalg

mat = random.normal(0.5, 0.1, size=[500, 500])

sym = mat * mat.T

cov = np.diag(np.diag(sym))

check = linalg.cholesky(cov)

print(cov)