# -*- coding: utf-8 -*-
from __future__ import division
import os
import copy
import sys
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier as SGD
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import BernoulliNB as BNB

from pandas import read_csv
import csv
import time

import warnings
warnings.filterwarnings("ignore")

def apply_inc_learner(learner, x , y, train, test):
       
        scores = []
        learner.partial_fit(x[train], y[train], classes = np.unique(y))
        score = learner.score(x[test], y[test])
        scores.append(score * 100)
        return scores

def apply_learner(learner, x , y, train, test):
        
        scores = []
        learner.fit(x[train], y[train])
        score = learner.score(x[test], y[test])
        scores.append(score * 100)
        return scores

# define the path of csv files
path = '...'
os.chdir(path)

d = {}
test_size = 0.1

datasets = []
for i in os.listdir(os.getcwd()):
    datasets.append(i)

# set the ids of the folds that lead to a different random_state of kFold, leading to len(folds) * 10-cross-validation processes
folds = [1, 2, 3, 4, 5, 7, 23, 66, 123, 2018]
# else, set just a seed into the folds list for one only 10-cross-validation procedure
folds = [23]

learners = [SGD(loss= 'log') ,  SGD(loss= 'modified_huber'), SGD(loss= 'log' , penalty = 'l1') , SGD(loss= 'log' , penalty = 'elasticnet') , SGD(loss= 'modified_huber' , penalty = 'l1') , SGD(loss= 'modified_huber' , penalty = 'elasticnet') , MNB(), BNB()]

for t in learners:
    
    l = []
    print '#### \t' , t, '\t ####' 
    for x in range(0, len(datasets)):

        lea = copy.deepcopy(t)
        acc = []
        stdev = []
        
        dataframe = read_csv(datasets[x] , skiprows = 1 , header=None)
        dataframe = dataframe.dropna()
        dataset = dataframe.values
        print
        features = int(dataset.shape[1]) - 1
        X = dataset[ : , 0 : features].astype('float64')
        Y = dataset[ : , features]
        
        start = time.time()
        cvscores = []
        for fold in folds:
            
            kfold = StratifiedKFold(n_splits=10, shuffle = True, random_state = fold)    
      
            for train, test in kfold.split(X, Y):

                    cvscores.append( apply_learner(lea, X, Y, train, test)[0] )

        end = time.time()
        toc = (end - start)
        
        print("dataset %s with shape %s : %.4f%% (+/- %.4f%%)" % ( datasets[x][0 : datasets[x].find('.')], str(dataframe.values.shape), np.mean(cvscores), np.std(cvscores)  ))

        acc.append(np.mean(cvscores))
        stdev.append(np.std(cvscores))
        l.extend([acc[0],stdev[0], toc])

    d[str(lea)[0 : str(lea).find(')')]] = l  
    print
    
with open('mycsvfile.csv', 'wb') as f: 
    w = csv.DictWriter(f, d.keys())
    w.writeheader()
    w.writerow(d)