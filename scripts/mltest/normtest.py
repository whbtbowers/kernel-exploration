
import numpy as np
import pandas as pd
import p2funcs as p2f

from scipy.stats import normaltest


inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',', header=None, index_col=0)

X_imp = p2f.filt_imp(inp_df, 0.1)

k2, p_vals = normaltest(X_imp)

alpha = 0.05
# Bonferroni correction
alpha_bon = alpha/len(p_vals)
n_normal = 0

for p in p_vals:
    if p > alpha_bon:
        n_normal += 1

print(n_normal)
#print('k2: %s' % str(k2))
#print('p: %s' % str(p))
#print(len(p))
