#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:16:11 2018

@author: whb17
"""
import p2funcs as p2f

tier1 = [0.002/10000, 0.2/1000, 0.2/100, 0.2/10, 0.2, 0.2*10]

tier2 = list(p2f.frange(tier1[2-1], tier1[2], tier1[2-1])) + list(p2f.frange(tier1[2], tier1[2+1], tier1[2]))

print(tier2)