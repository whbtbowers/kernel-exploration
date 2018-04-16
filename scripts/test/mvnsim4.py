#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:36:05 2018

@author: whb17
"""

from scipy.stats import multivariate_normal
x = np.linspace(0, 5, 20, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)

print(y)