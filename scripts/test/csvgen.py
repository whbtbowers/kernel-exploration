'''
import random
import numpy as np
cols = 5
rows = 8
#print('number of columns is %d, number of rows is %d' % (cols, rows))
#print(f'number of columns is {cols}, number of rows is {rows}')

#for x in range(10):
#    print(random.randint(1,101))

mat = np.array

for i in range(cols):
    print(random.randint(1,101))

#simcsv = open('test.csv', 'w')
'''

import csv
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'WonderfulSpam'])
