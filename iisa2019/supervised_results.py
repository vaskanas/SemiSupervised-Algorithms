import sys
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy import stats
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import ExtraTreesClassifier as EXT
from sklearn.neural_network import MLPClassifier as MLP

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import os

import warnings
warnings.filterwarnings("ignore")

def download1(decide):
    
    os.chdir('give your pathe with the datasets here')
    
    if decide == 1:
        data = pd.read_csv('voice_numeric.csv')
        data.columns.values[-1] = 'Class'
    elif decide == 2:
        data = pd.read_csv('ANAD_Normalized.csv')
        data = data.iloc[:, 2:]
        data.columns.values[0] = 'Class'
    else:
        data = pd.read_csv('X_y_labeled_data_hate_speech.csv')
        data = data.iloc[:, 1:] #drop the id column
        data.columns.values[-1] = 'Class'
    
    
    data = shuffle(data, random_state = 1)
    
    X = data.astype('float64')
    y = data.Class
    
    print ('given dataset: ', X.shape, y.shape)
    return (X, y)

decide = 3
(X, y) = download1(decide) # DATAFRAME , SERIES

if decide == 1:
    X = X.iloc[:,:-1]
elif decide == 2:
    X = X.iloc[:,1:]
else:
    X = X.iloc[:,:-1]
#%%

def supervised_stratified(x_tr, y_tr, x_ts, y_ts, lea):
    acc = []
    f1 = []
    for i in range(0, len(y_ts)):
        lea.fit(x_tr[i], y_tr[i])
        prec, rec, ff1, sup = precision_recall_fscore_support(y_ts[i], lea.predict(x_ts[i]), average='weighted')
        acc.append(lea.score(x_ts[i] ,y_ts[i]) * 100)
        f1.append(ff1 * 100)
    #print acc
    return np.mean(acc) , np.std(acc), np.mean(f1) , np.std(f1)

rf_super   = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', random_state = 23) #'balanced_subsample')
ext_super  = EXT(n_estimators=100, class_weight = 'balanced', random_state = 23)
nb_super   = NB()
knn_super  = KNN()
mlp_super  = MLP(random_state = 23)

rf_upper_bound   = np.mean(cross_val_score(  rf_super, X, y, cv = 3)) * 100
nb_upper_bound   = np.mean(cross_val_score(  nb_super, X, y, cv = 3)) * 100
knn_upper_bound  = np.mean(cross_val_score( knn_super, X, y, cv = 3)) * 100
ext_upper_bound  = np.mean(cross_val_score( ext_super, X, y, cv = 3)) * 100
mlp_upper_bound  = np.mean(cross_val_score( ext_super, X, y, cv = 3)) * 100

sss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.1, random_state = 23)
x_tr, y_tr, x_ts, y_ts = [], [], [], []
for train_index, test_index in sss.split(X, y):
    
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        x_tr.append(X_train)
        y_tr.append(y_train)
        x_ts.append(X_test)
        y_ts.append(y_test)

rf_acc  , rf_std, rf_f1, rf_f1_std  = supervised_stratified(x_tr, y_tr, x_ts, y_ts, rf_super)
ext_acc , ext_std, ext_f1, ext_f1_std = supervised_stratified(x_tr, y_tr, x_ts, y_ts, ext_super)
nb_acc  , nb_std, nb_f1, nb_f1_std  = supervised_stratified(x_tr, y_tr, x_ts, y_ts, nb_super)
knn_acc , knn_std, knn_f1, knn_f1_std = supervised_stratified(x_tr, y_tr, x_ts, y_ts, knn_super)
mlp_acc , mlp_std, mlp_f1, mlp_f1_std = supervised_stratified(x_tr, y_tr, x_ts, y_ts, mlp_super)


print('---> supervised acc: ')
print('rf: '  , rf_upper_bound,  '%')
print('nb: '  , nb_upper_bound,  '%')
print('knn: ' , knn_upper_bound, '%')
print('ext: ' , ext_upper_bound, '%')
print('mlp: ' , mlp_upper_bound, '%')

#save the session

#import dill
#filename = input('give the desired name: ') + '.pkl' # example : 'hate_supervised.pkl'
#dill.dump_session(filename)

# and to load the session again:
#dill.load_session(filename)


