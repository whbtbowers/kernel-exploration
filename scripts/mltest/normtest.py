import numpy as np
import pandas as pd
import p2funcs as p2f

from scipy.stats import normaltest, shapiro


inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_UnTransformed_Data.txt', sep='\t', header=None, index_col=0)

X_imp = p2f.filt_imp(inp_df, 0.1)

#print(X_imp)
## Test using normaltest ##
k2, nt_p_vals = normaltest(X_imp)

alpha = 0.05
nt_n_normal = 0

for p in nt_p_vals:
    if p > alpha:
        nt_n_normal += 1

## Test using shapiros

w_vals = []
sh_p_vals = []

#for i in X_imp.columns:
w, p = shapiro(X_imp)
    #w_vals.append(w)
    #sh_p_vals.append(p)

sh_n_normal = 0

# for p in sh_p_vals:
#     if p > alpha:
#         nt_n_normal += 1

print('Normal test result: %s' % (nt_n_normal))
print('Shapiro result: %s' % (p))
