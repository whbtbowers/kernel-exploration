"""
KPCA using method in mlextend tutorial
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import RBFKernelPCA as KPCA

#import subset of initial csv
inp_csv = pd.read_csv('../../sampdata/rand100subset3.csv', delimiter=',', header=0)

#plot based on vector
#input vector (eg glucose & bmi)
#category (eg sex)
#out_type - output type, 'show' or 'save'

#chosen dimensions
glucose = inp_csv['glucose']
sex = inp_csv['sex']
bmi = inp_csv['bmi']

'''
print('glucose')
print(glucose)
print(len(glucose))

print('bmi')
print(bmi)
print(len(bmi))

print('sex')
print(sex)
print(len(sex))
'''

#for i in range(len(sex)):
    #print(sex[i])
'''
#create individual array for each parameter
#sort glucose and bmi by sex
cat1glu = []
cat1bmi = []
cat2glu = []
cat2bmi = []

for i in range(len(sex)):
    if sex[i] == 1:
        cat1glu.append(glucose[i])
        cat1bmi.append(bmi[i])
    if sex[i] == 2:
        cat2glu.append(glucose[i])
        cat2bmi.append(bmi[i])
'''
#print('cat1glu')
#print(cat1glu)
#print('cat1bmi')
#print(cat1bmi)
#print('cat2glu')
#print(cat2glu)
#print('cat2bmi')
#print(cat2bmi)

'''
#plot from individual parameters
#Sex 1 glucose v BMI
sex1 = plt.scatter(cat1bmi,
    cat1glu,
    color ='magenta',
    marker='s',     #square marker
    alpha=0.5,
    )
#Sex 2 glucose v BMI
sex2 = plt.scatter(cat2bmi,
    cat2glu,
    color='cyan',
    marker='p',     #pentagon marker
    alpha=0.5,
    )


plt.title('BMI vs glucose by sex')
plt.ylabel('Serum glucose concentration')
plt.xlabel('BMI')

plt.legend([sex1, sex2], ['Sex 1', 'Sex 2'])
#plt.show()
plt.savefig('../../figs/bivariate/subsetkpca1_1')
plt.close()
'''



#Create empty array to add vectors to
data = []

for i in range(len(bmi)):

    #create sample vector
    xi = []
    xi.append(bmi[i])
    xi.append(glucose[i])

    #Append sample vector to array
    data.append(xi)

#simple way to convert to numpy array
X = np.array(data)
#print(X)

"""

#generate graph from matrix
for i in range(len(X)):
    if X[i][2] == 1:
        #Sex 1 glucose v BMI
        sex1 = plt.scatter(X[i][0], #bmi
            X[i][1],    #glucose
            color ='red',
            marker='o',     #circle marker
            alpha=0.5,
            )
    elif X[i][2] == 2:
        sex2 = plt.scatter(X[i][0], #bmi
            X[i][1],    #glucose
            color='blue',
            marker='^',     #triangle marker
            alpha=0.5,
            )
plt.title('BMI vs glucose by sex')
plt.ylabel('Serum glucose concentration')
plt.xlabel('BMI')
plt.legend([sex1, sex2], ['Sex 1', 'Sex 2'])

#plt.show()
plt.savefig('../../figs/bivariate/subsetkpca1_2')
plt.close()
"""

#PCA only takes 2D axis. Look into how to deal with that later.
pca = PCA(n_components=2) #2-component linear PCA
X_pca = pca.fit(X).transform(X)

#print(X_pca)
'''
#Graph after pca
#generate graph from matrix
for i in range(len(X)):
    if sex[i] == 1:
        #Sex 1 glucose v BMI
        sex1 = plt.scatter(X_pca[i][0], #bmi
            X_pca[i][1],    #glucose
            color ='red',
            marker='o',     #circle marker
            alpha=0.5,
            )
    elif sex[i] == 2:
        sex2 = plt.scatter(X_pca[i][0], #bmi
            X_pca[i][1],    #glucose
            color='blue',
            marker='^',     #triangle marker
            alpha=0.5,
            )
plt.title('BMI vs glucose by sex after linear PCA')
plt.ylabel('Serum glucose concentration')
plt.xlabel('BMI')
plt.legend([sex1, sex2], ['Sex 1', 'Sex 2'])

#plt.show()
plt.savefig('../../figs/bivariate/subsetkpca1_3')
plt.close()
'''

#kpca with arbitrary gamma
gamma = 500
kpca = KPCA(gamma=gamma, n_components=2)
#Fit X with above KPA specifications
kpca.fit(X)
#Project X values onto 'new' (higher dimensional?) feature space (rep by 'g' in associated notes)
X_kpca = kpca.X_projected_

#Graph after kpca
#generate graph from matrix
for i in range(len(X)):
    if sex[i] == 1:
        #Sex 1 glucose v BMI
        sex1 = plt.scatter(X_kpca[i][0], #bmi
            X_kpca[i][1],    #glucose
            color ='red',
            marker='o',     #circle marker
            alpha=0.5,
            )
    elif sex[i] == 2:
        sex2 = plt.scatter(X_kpca[i][0], #bmi
            X_kpca[i][1],    #glucose
            color='blue',
            marker='^',     #triangle marker
            alpha=0.5,
            )

plt.title('BMI vs glucose by sex after kernel PCA (gamma=' + str(gamma) + ')')
plt.ylabel('Serum glucose concentration')
plt.xlabel('BMI')
plt.legend([sex1, sex2], ['Sex 1', 'Sex 2'])

#plt.show()
plt.savefig('../../figs/bivariate/subsetkpca1_4_gamma' + str(gamma) + '.png')
plt.close()
