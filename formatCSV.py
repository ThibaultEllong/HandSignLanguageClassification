import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

dataframe = pd.read_csv('./Dataset/CassiopéeShift/Activité.csv', sep=r'\s*,\s*', engine='python')

#for i in dataframe.columns:
    #print(dataframe[i][0])
#dataframe.iloc[0] = dataframe.iloc[0].shift(1)
#dataframe.to_csv('./Dataset/Cassiopée/900Test.csv', index=False)

def shiftCSV(file):
    dataframe = pd.read_csv('./Dataset/Cassiopée/' + file, sep=r'\s*;\s*', engine='python')
    dataframe.iloc[0] = dataframe.iloc[0].shift(1)
    dataframe.to_csv('./Dataset/CassiopéeShift/' + file, index=False)

def normalize(file):
    
    dataframe = pd.read_csv('../Dataset/CassiopéeShift/' + file, sep=r'\s*,\s*', engine='python')
    columns = dataframe.columns.tolist()
    print(columns)
    norm_vals_X = dataframe['pelvis_T_glob'][1:]
    norm_vals_Y = dataframe['pelvis_T_glob.1'][1:]
    norm_vals_Z = dataframe['pelvis_T_glob.2'][1:]

    for j in range(1, len(dataframe.index)):
        for i in dataframe.columns:
            if dataframe[i][0] == 'X':
                dataframe[i][j] = round(float(dataframe[i][j]) - float(norm_vals_X[j]), 3) #convert string of int to int
            elif dataframe[i][0] == 'Y':
                dataframe[i][j] = round(float(dataframe[i][j]) - float(norm_vals_Y[j]), 3)
            elif dataframe[i][0] == 'Z':
                dataframe[i][j] = round(float(dataframe[i][j]) - float(norm_vals_Z[j]), 3)
        print('Done : ' , file , '. Only ' , len(dataframe.index) - j , ' to go !')
    dataframe.to_csv('./Dataset/CassiopéeNorm/' + file, index=False)
                
for file in os.listdir('./Dataset/CassiopéeShift/'):
   if file.endswith('.csv'):
        normalize(file)
        
 

def selectColumn(file, str):
    return 0

