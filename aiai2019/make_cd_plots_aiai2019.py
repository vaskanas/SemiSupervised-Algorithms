# -*- coding: utf-8 -*-
import os  
from pandas import read_excel
import numpy as np
import pandas as pd
import Orange
import matplotlib.pyplot as plt

os.chdir('...') # directory that contains Statstical rankings.xlsx file

sheets = ['log_l2','mhuber_l2','log_l1','mhuber_l1','log_elnet','mhuber_elnet']

lr = ['5','','15','','25']

for i in sheets:

    df = pd.read_excel('binary_ranking.xlsx', sheet_name=i)
    for j in range(0, df.shape[1],2):

            names = df.iloc[:,j]
            avranks = df.iloc[:,j+1]
            for k in range(0,names.shape[0]):
                
                    if 'inc' in names[k]:
                        names[k] = names[k][0:names[k].find('inc')-1] + ')'
                        names[k] = names[k][0:names[k].find('(')] + 'inc' + names[k][names[k].find('(') :]
                    elif 'conv' in names[k]:
                        names[k] = names[k][0:names[k].find('conv')-1] + ')'
                        names[k] = names[k][0:names[k].find('(')] + 'batch' + names[k][names[k].find('(') :]
                    else:
                        print('error')
                    
                    if 'super' in names[k]:
                        names[k] = 'Super (' + names[k][0:names[k].find('super')-1] + ')'
                    else :
                        names[k] = 'AL (' + names[k] + ')'
                            

            cd = Orange.evaluation.compute_CD(avranks, 190 , alpha = '0.05', test = 'bonferroni-dunn') 
            print(cd)
            plt.figure(figsize=(9, 9))
        
            Orange.evaluation.graph_ranks(avranks, names, cd = cd, width = 15, textspace = 5)
            plt.savefig("SGD( " + i + ') R = ' + lr[j] + '%.png', dpi=600) 