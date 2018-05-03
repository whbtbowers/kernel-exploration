
import pandas as pd
#header=0 makes first line header
df = pd.read_csv('../../data/MESA_Clinical_data_(full_COMBI-BIO).xlsx - COMBI-BIO Merged Clinical data.csv', delimiter=',', header=0)

#values in table
#print(df.values)

#number of rows
#print(len(df.index))

#headers
print(df.columns.values)
