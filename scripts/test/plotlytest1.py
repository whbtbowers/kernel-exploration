#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:07:21 2018

@author: whb17
"""

import pandas as pd
import numpy as np
import plotly as py
import plotly.plotly as ply
import plotly.graph_objs as go

py.tools.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = go.Data([trace0, trace1])

ply.iplot(data, filename = 'basic-line')