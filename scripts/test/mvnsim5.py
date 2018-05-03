#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:30:52 2018

@author: whb17
"""

from sklearn.mixture import GaussianMixture

nd_mvn = GaussianMixture(n_components=1, covariance_type='diag', means_init=[60, 50])

print(nd_mvn)