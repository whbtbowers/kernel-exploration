
import numpy as np
import pandas as pd
import p2funcs as p2f

from scipy.stats import normaltest, shapiro


inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',', header=None, index_col=0)




X_imp = p2f.filt_imp(inp_df, 0.1)

## Test using normaltest ##
k2, nt_p_vals = normaltest(X_imp)

alpha = 0.05
# Bonferroni correction
nt_alpha_bon = alpha/len(nt_p_vals)
nt_n_normal = 0

for p in nt_p_vals:
    if p > nt_alpha_bon:
        nt_n_normal += 1

## Test using shapiros

w_vals = []
sh_p_vals = []

for i in X_imp.columns:

    w, p = shapiro(X_imp[i])
    w_vals.append(w)
    sh_p_vals.append(p)

# Bonferroni correction
sh_alpha_bon = alpha/len(sh_p_vals)
sh_n_normal = 0

for p in sh_p_vals:
    if p > sh_alpha_bon:
        nt_n_normal += 1

print(sh_n_normal)
