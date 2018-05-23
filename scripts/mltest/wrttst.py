#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:34:47 2018

@author: whb17
"""
import numpy as np
import p2funcs as p2f

title1 = 'list1\n'
list1 = np.array([3, 4, 5, 5, 4, 2])

title2 = '\nlist2\n'
list2 = np.array([6, 44, 3, 6, 4, 3])

p2f.writetext(title1, 'test.txt', '../../data/plotting/')
p2f.writetext(list1.tolist(), 'test.txt', '../../data/plotting/')
p2f.writetext(title2, 'test.txt', '../../data/plotting/')
p2f.writetext(list2.tolist(), 'test.txt', '../../data/plotting/')