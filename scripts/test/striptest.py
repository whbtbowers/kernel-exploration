#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:16:13 2018

@author: whb17
"""

class MyStr(str):
    def __repr__(self):
        return super(MyStr, self).__repr__().strip("'")

s1 = 'hello\nworld'
s2 = MyStr(s1)

slist = [s1, s2]

print("s1: %r" % s1)
print("s2: %r" % s2)

print(slist)