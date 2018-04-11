import pandas as pd
import random as ran

#generate array of random numbers of given length
#numbers range from 0 to length of matrix
def randnums(totalrows, numrowswanted):
    nums = []

    for i in range(numrowswanted):
        nums.append(ran.randint(0,totalrows))
    return(nums)

def csv_subsetter(outp_filename,inp_csv_path, numrowswanted):

    #Header=0 makes first line header
    #Assumes variables seperated by comma.
    inp_csv = pd.read_csv(inp_csv_path, delimiter=',', header=0)
    #number of rows in csv
    nrows = len(inp_csv.index)
    # get rows given by random numbers
    getrows = inp_csv.iloc[randnums(nrows, numrowswanted)]
    #write subset table to textfile
    getrows.to_csv(path_or_buf=outp_filename, header=True, sep=',')

# Create subset table of 100 random rows
csv_subsetter('../sampdata/rand100subset3.csv', '../data/MESA_Clinical_data_(full_COMBI-BIO).xlsx - COMBI-BIO Merged Clinical data.csv', 100)
