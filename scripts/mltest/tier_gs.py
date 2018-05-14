#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:04:39 2018

@author: whb17
"""

import p2funcs as p2f

k_list = [2e-7, 0.000002, 0.00002, 0.0002, 0.002, 0.02, 0.2, 2.0]

select_k = 0.0002

k_i = k_list.index(select_k)


k_list_t2=list(p2f.frange(k_list[k_i], k_list[k_i+1], k_list[k_i]))

print(k_list_t2)