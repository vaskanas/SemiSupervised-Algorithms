# -*- coding: utf-8 -*-

from __future__ import division
import libact
import os
import copy
import sys
sys.path.append(r'/home/user/anaconda2/lib/python2.7/site-packages')

import numpy as np
import matplotlib.pyplot as plt
try:
	from sklearn.model_selection import train_test_split
except ImportError:
	from sklearn.cross_validation import train_test_split
import xlwt

from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import BernoulliNB as BNB
from numpy.testing import assert_array_equal

#from libact.models import SklearnAdapter, SklearnProbaAdapter
from libact.base.interfaces import Model, ContinuousModel, ProbabilisticModel
from libact.utils import inherit_docstring_from, zip
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from libact.base.dataset import Dataset
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.base.interfaces import QueryStrategy
from libact.query_strategies import ActiveLearningByLearning

from pandas import read_csv
from decimal import *
getcontext().prec = 4

import timeit
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/user/Downloads/binary csvs')
#os.chdir('...') #path with csvs of binary datasets

# Getting all the arff files from the current directory
l_csv = [x for x in os.listdir('.') ]

test_size = 0.10
times_l = []
print l_csv


def run(trn_ds, tst_ds, lbr, model, qs, quota, j):
	
	E_in, E_out, l = [], [], []
	model.train(trn_ds)
	E_out.append(1 - model.score(tst_ds))
	counter = 0
	steps = 15
	k = int(j / steps)
	
	if k == 0:
		k = 1
	
	for _ in range(0,steps):
		counter = counter + 1   
		l = []
		
		for i in range(0,k):
			ask_id = qs.make_query()
			l.append(ask_id)
			X, _ = zip(*trn_ds.data)
			lb = lbr.label(X[ask_id])
			trn_ds.update(ask_id, lb)

		model.train(trn_ds)
		E_out = np.append(E_out, 1 - model.score(tst_ds))
		#print '-->', len(E_out)
	return E_out, len(E_out), trn_ds, k, j


def split_train_test(test_size, n_labeled, myfold, myflag, random_col, name, path):

	dataframe = read_csv(path + name, skiprows = 1 , header=None)
	dataframe = dataframe.dropna()
	dataset = dataframe.values

	features = dataset.shape[1] - 1
	X = dataset[:, 0:features].astype(float)
	y = dataset[:,features]
	
	instances = len(y)
	if myflag == 0: 
		return instances
	else:
		sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = myfold)
		l_train = []
		l_test = []
		
		for train_index, test_index in sss.split(X, y):
			l_train.append(train_index)
			l_test.append(test_index)
	
		if myflag == 1:
			X_train = X[l_train] 
			y_train = y[l_train]
			X_train,  y_train = shuffle(X_train, y_train, random_state = random_col)
			return X_train, y_train
		else:
			X_train = X[l_train] 
			y_train = y[l_train]
			X_test  = X[l_test] 
			y_test  = y[l_test]
			X_train, y_train = shuffle(X_train, y_train, random_state = random_col)
	 
	 
		trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))  
		tst_ds = Dataset(X_test, y_test)
		fully_labeled_trn_ds = Dataset(X_train, y_train)

	return trn_ds, tst_ds, y_train, fully_labeled_trn_ds, instances 

def main():
 
 global count, path_csv, test_size
 path_csv = '' 
 random_shuffle_id = 23

 for file_csv in l_csv:
	book = xlwt.Workbook(encoding="utf-8")
	start = datetime.now()
	folds = [1, 2]#, 3, 4, 5, 7, 23, 66, 123, 2018]
   
	for fold in folds:
		message = "Sheet " + str(fold)
		sheet1 = book.add_sheet(message)
			
		SIZE = (1 - test_size) * split_train_test(test_size, 1, fold, 0, random_shuffle_id, file_csv, path_csv)
		count = -1
		
		for col in range(1,2): #we could increase the second argument of range, in case that more we would like to run the experiment again for the same fold with different shuffle e.g. 5x2 evaluation

			print '***********file*********** = ', file_csv
			print '***********col************ = ', col
			print '***********fold*********** = ', fold
			print 'SIZE of L + U              = ', int(SIZE)
			print
			
			myspace = np.linspace(int(0.05 * SIZE), int(0.25 * SIZE) + 1, 3)
			learners = [SGD(loss= 'log') ,  SGD(loss= 'modified_huber'), SGD(loss= 'log' , penalty = 'l1') , SGD(loss= 'log' , penalty = 'elasticnet') , SGD(loss= 'modified_huber' , penalty = 'l1') , SGD(loss= 'modified_huber' , penalty = 'elasticnet')]
			
			for lea in learners:
			  
			  counter_j = -1
			  counter_jj = -1
			  count = count + 1
			  my_clf =  lea
			  print str(my_clf)[0: str(my_clf).find('(')] + '(' + str(my_clf)[str(my_clf).find('loss') : str(my_clf).find(',', str(my_clf).find('loss'))] + ')' + ' ,(' + str(my_clf)[str(my_clf).find('penalty') : str(my_clf).find(',', str(my_clf).find('penalty'))] + ')'

			  for j in myspace:

					j = int(round(j))
					counter_j = counter_j + 1 
					n_labeled = j  # number of samples that are initially labeled
					print '**** Labeled instances = ', j

					metrics = ['lc', 'entropy' , 'sm' , 'random'] 
					
					for jj in metrics:
						
						trn_ds, tst_ds, y_train, fully_labeled_trn_ds, initial_instances = split_train_test(test_size, n_labeled, fold, random_shuffle_id, col, file_csv, path_csv)
						trn_ds2 = copy.deepcopy(trn_ds)
						lbr = IdealLabeler(fully_labeled_trn_ds)
						train_data = int(initial_instances - initial_instances * test_size)
						quota = len(y_train) - n_labeled    # number of samples to query
						
						# Comparing UncertaintySampling strategy with RandomSampling.                        
						counter_jj = counter_jj + 1

						if jj != 'random' :

							print '**** Metric of Uncertainty Sampling strategy = ', jj
							qs1 = UncertaintySampling (trn_ds, kernel = jj , model =  SklearnProbaAdapter(my_clf))
							model = SklearnProbaAdapter(my_clf)
							E_out_1, ttt , trn_ds_returned , aa , bb = run(trn_ds, tst_ds, lbr, model, qs1, quota, j)
							
						else:

							print '**** Baseline Sampling strategy = ', jj
							qs1 = RandomSampling (trn_ds, model =  SklearnProbaAdapter(my_clf))
							model = SklearnProbaAdapter(my_clf)
							E_out_1, ttt , trn_ds_returned , aa , bb = run(trn_ds, tst_ds, lbr, model, qs1, quota, j)
							
						#print '#(L+U) = ' , len(trn_ds_returned) , ' instances per iter = ', aa, ' initial_amount = ', bb , ' #L = ', trn_ds_returned.len_labeled(), ' #U = ', trn_ds_returned.len_unlabeled()
						
						if count != 0:
						  down_cells = len(E_out_1) + 9
						else:
							down_cells = 0

						i = 8 + down_cells * count 
						
						sheet1.write(i - 7, counter_jj + counter_j, jj)                                  # metric of incertaintly
						sheet1.write(i - 6, counter_jj + counter_j, quota)                               # amount of U
						sheet1.write(i - 5, counter_jj + counter_j, aa)                                  # instanes inserted per iteration
						sheet1.write(i - 4, counter_jj + counter_j, bb)                                  # amount of L
						sheet1.write(i - 3, counter_jj + counter_j, trn_ds_returned.len_labeled())       # amount of training data after active learning procedure
						sheet1.write(i - 2, counter_jj + counter_j, trn_ds_returned.len_unlabeled())     # amount of unlabeled instances after active learning procedure
						
						sheet1.write(i-8, counter_jj + counter_j, str(my_clf)[0: str(my_clf).find('(')] + '(' + str(my_clf)[str(my_clf).find('loss') : str(my_clf).find(',', str(my_clf).find('loss'))] + ')' + ' ,(' + str(my_clf)[str(my_clf).find('penalty') : str(my_clf).find(',', str(my_clf).find('penalty'))] + ')')
						for n in E_out_1:

							sheet1.write(i, counter_jj + counter_j, n)
							i = i+1  
						#print 'error in last iteration: ', E_out_1[-1]
						print						
	print("> Compilation Time : %s", (datetime.now() - start).total_seconds())                  
	print("AIAIexperiment_" + file_csv[0:-4] +  ".xls")              
	book.save("AIAIexperimetn_" + file_csv[0:-4] + "_incremental_" + str(fold) +  ".xls")
	
	times_l.append((datetime.now() - start).total_seconds())
	
if __name__ == '__main__':
	
	getcontext().prec = 4
	main()
	print(l_csv)
	print(times_l)