import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as GraB 
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.gaussian_process import GaussianProcessClassifier as GPcl
from sklearn.ensemble import ExtraTreesClassifier as EXT
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import precision_recall_fscore_support


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import os


import warnings
warnings.filterwarnings("ignore")


def download_dataset(dataset):
    
    os.chdir("provide path")
    if dataset == 1:
        data = pd.read_csv('voice_numeric.csv')
        data.columns.values[-1] = 'Class'
    elif dataset == 2:
        data = pd.read_csv('ANAD_Normalized.csv')
        data = data.iloc[:, 2:]
        data.columns.values[0] = 'Class'
    else:
        
        data = pd.read_csv('X_y_labeled_data_hate_speech.csv')
        data = data.iloc[:, 1:] 
        data.columns.values[-1] = 'Class'
    
    
    data = shuffle(data, random_state = 1)
    
    X = data.astype('float64')
    y = data.Class
    print ('given dataset: ', X.shape, y.shape)
    return (X, y)   

def split(train_size):
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_full, y_train_full, X_test, y_test)

def split_alssl(X, y, repeats, testsize):
    
    sss = StratifiedShuffleSplit(n_splits = repeats, test_size = testsize, random_state = 23)
    x_tr, y_tr, x_ts, y_ts = [], [], [], []
    for train_index, test_index in sss.split(X, y):
        
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        x_tr.append(X_train)
        y_tr.append(y_train)
        x_ts.append(X_test)
        y_ts.append(y_test)
    return x_tr, y_tr, x_ts, y_ts 

class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass


class RfModel(BaseModel):

    model_type = 'Random Forest'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        #print ('training random forest...')
        self.classifier = RandomForestClassifier(n_estimators=100, class_weight=c_weight, random_state = 23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
    
class ExtModel(BaseModel):

    model_type = 'Extra Trees'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        
        self.classifier = EXT(n_estimators=100, class_weight=c_weight, random_state = 23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
    
class NBModel(BaseModel):

    model_type = 'NB'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        
        self.classifier = NB()
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
      
class KNNModel(BaseModel):

    model_type = 'KNN'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        
        self.classifier = KNN()
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
    
class MLPModel(BaseModel):

    model_type = 'MLP'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        
        self.classifier = MLP(random_state = 23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
    
class TrainModel:

    def __init__(self, model_object):        
        self.accuracies = []
        self.model_object = model_object()
        self.precision = []
        self.recall = []
        self.fscore =[]        

    def print_model_type(self):
        print (self.model_object.model_type)


    def train(self, X_train, y_train, X_val, X_test, c_weight):
       
        t0 = time.time()
        (X_train, X_val, X_test, self.val_y_predicted,
         self.test_y_predicted) = \
            self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_test)  


    def get_test_accuracy(self, i, y_test, message_AL_SSL):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        prec, rec, f1, sup = precision_recall_fscore_support(y_test, self.test_y_predicted, average='weighted')
        self.accuracies.append(np.round(classif_rate,3))
        self.precision.append(np.round(prec*100,3))
        self.recall.append(np.round(rec*100,3))
        self.fscore.append(np.round(f1*100,3))               

        
class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
        return selection


class EntropySelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection
      
      
class MarginSamplingSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection

class MinStdSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        std = np.std(probas_val * 100, axis=1)
        selection = std.argsort()[:initial_labeled_samples]
        selection = selection.astype('int64')

        return selection
    
class SSLselection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, toquery):
        maxInRows = np.amax(probas_val, axis=1) #maximum per row/instance
        maxInPos = []
        for i in range(0, probas_val.shape[0]):
            maxInPos.append(np.where(probas_val[i,:] == maxInRows[i])[0][0])
        e = sorted(range(len(maxInRows)),key = maxInRows.__getitem__)
        
        selection = [e[-1 * toquery : ]]

        return selection
    
class Normalize(object):
    
    def normalize(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)
        return (X_train, X_val, X_test) 
    
    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
        X_test  = self.scaler.inverse_transform(X_test)
        return (X_train, X_val, X_test)
      
def get_k_random_samples(initial_labeled_samples, X_train_full,
                         y_train_full):
    permutation = np.random.choice(trainset_size,
                                   initial_labeled_samples,
                                   replace=False)
    print ()
    print ('initial random chosen samples', permutation.shape),

    X_train = X_train_full.iloc[permutation].values
    y_train = y_train_full.iloc[permutation].values
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    return (permutation, X_train, y_train)


def get_k_random_samples_stratified(initial_labeled_samples, X_train_full, y_train_full, LR):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 1 - LR, random_state = 23)
    for tr, ts in sss.split(X_train_full, y_train_full):
        X_train = X_train_full.loc[tr]
        y_train = y_train_full.loc[tr]
    permutation = np.array(tr)
    print ()
    print ('initial random chosen samples', permutation.shape),
    X_train = X_train_full.iloc[permutation].values
    y_train = y_train_full.iloc[permutation].values
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    return (permutation, X_train, y_train)

class TheAlgorithm(object):

    accuracies = []
    precision, recall, fscore = [], [], []
    
    def __init__(self, initial_labeled_samples, model_object, selection_function, L0, selection_function_ssl, ssl_ratio, permutation, X_train, y_train):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function
        self.L0 = L0
        self.sample_selection_function_ssl = selection_function_ssl
        self.ssl_ratio = ssl_ratio
        #self.repeats = repeats
        self.permutation = permutation
        self.X_train = X_train
        self.y_train = y_train

    def run(self, X_train_full, y_train_full, X_test, y_test, permutation, X_train, y_train):

       
        self.queried = 0 
        self.samplecount = [self.initial_labeled_samples]

        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X_train_full)
        y_val = np.copy(y_train_full)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.delete(y_val, permutation, axis=0)
        
        self.clf_model = TrainModel(self.model_object)
        (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
        active_iteration = 0
        self.clf_model.get_test_accuracy(active_iteration, y_test, 'Initial stage')

        while self.queried < max_queried:

            active_iteration += 1

            # get validation probabilities

            probas_val = \
                self.clf_model.model_object.classifier.predict_proba(X_val)

            # select samples using a selection function

            uncertain_samples = \
                self.sample_selection_function.select(probas_val, self.initial_labeled_samples) 

            # get the uncertain samples from the validation set

            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))
            

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
                          

            self.queried += self.initial_labeled_samples
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
            self.clf_model.get_test_accuracy(active_iteration, y_test, 'Active Learning')
            
            #print('----------ssl starts----------')
            probas_val_ssl = \
                self.clf_model.model_object.classifier.predict_proba(X_val)
           
            uncertain_samples_ssl = \
                self.sample_selection_function_ssl.select(probas_val_ssl, self.initial_labeled_samples * self.ssl_ratio)
            
            
            X_train = np.concatenate((X_train, X_val[uncertain_samples_ssl]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples_ssl]))
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))

            X_val = np.delete(X_val, uncertain_samples_ssl, axis=0)
            y_val = np.delete(y_val, uncertain_samples_ssl, axis=0)
            print 

                      

            self.queried += (self.initial_labeled_samples * self.ssl_ratio)
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
            self.clf_model.get_test_accuracy(active_iteration, y_test, 'Semi-supervised Learning')
                
        print ('final active learning accuracies',
               self.clf_model.accuracies)
        
        return X_train, y_train, X_val, y_val, X_test, y_test


  
def experiment(d, models, selection_functions, Ks, repeats, contfrom, L0, selection_functions_ssl, ssl_ratio, X_train_full, y_train_full, X_test, y_test, permutation, X_train, y_train):
    
    print ('stopping at:', max_queried)
    count = 0
    for model_object in models:
      if model_object.__name__ not in d:
          d[model_object.__name__] = {}
      
      for selection_function in selection_functions:
        if selection_function.__name__ not in d[model_object.__name__]:
            d[model_object.__name__][selection_function.__name__] = {}
        
        for selection_function_ssl in selection_functions_ssl:
            if selection_function_ssl.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__][selection_function_ssl.__name__] = {}
        
            for k in Ks:
                d[model_object.__name__][selection_function.__name__][selection_function_ssl.__name__][str(k)] = []           
                
                for i in range(0, repeats):
                    count += 1
                    print('--> number of models' ,count)
                    X_train_full_exp = X_train_full[i]
                    y_train_full_exp = y_train_full[i]
                    X_test_exp = X_test[i]
                    y_test_exp = y_test[i]
                    if count >= contfrom:
                        print ('Count = %s, using model = %s, selection_function = %s, selection_function_ssl = %s, k = %s, iteration = %s., initial L0 set = %d.' % (count, model_object.__name__, selection_function.__name__, selection_function_ssl.__name__, k, i, L0))
                        alg = TheAlgorithm(k, 
                                           model_object, 
                                           selection_function,
                                           L0,
                                           selection_function_ssl,
                                           ssl_ratio,
                                           permutation, X_train, y_train
                                           )
                        
                        mytest = alg.run(X_train_full_exp, y_train_full_exp, X_test_exp, y_test_exp, permutation, X_train, y_train)
                        d[model_object.__name__][selection_function.__name__][selection_function_ssl.__name__][str(k)].append([alg.clf_model.accuracies, alg.clf_model.precision, alg.clf_model.recall, alg.clf_model.fscore])
                        
                        print ()
                        print ('---------------------------- FINISHED ---------------------------')
                        print ()
    return d, mytest


#%%
for LR in [0.01, 0.05]: 
    ssl_ration = [3, 4]
    query_pools = [160, 200]
    
    (X, y) = download_dataset(1)
    X = X.iloc[:,1:]
    
    dataset_size = X.shape[0]
    testset_size = 0.1 * dataset_size
    repeats = 3
    (X_train_full, y_train_full, X_test, y_test) = split_alssl(X, y, repeats, testset_size / dataset_size)
    print ('L + U:', X_train_full[0].shape, y_train_full[0].shape)
    print ('test :', X_test[0].shape, y_test[0].shape)
    classes = len(np.unique(y))
    print ('unique classes', classes)
    trainset_size = X_train_full[0].shape[0]
    L0 = int(trainset_size * LR)    
        
    for pointer in range(0, len(ssl_ration)):
        (permutation, X_train, y_train) = get_k_random_samples_stratified(L0, X_train_full[pointer], y_train_full[pointer], LR)
        ssl_ratio = ssl_ration[pointer]
        max_queried = query_pools[pointer]
    
        d = {}
        stopped_at = -1 
        
        Ks_str = [ '2', '5', '10', '20' ] 
        Ks     = [  2 ,  5 ,  10 ,  20  ]
        
        
        selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection, MinStdSelection]
        selection_functions_str = ['RandomSelection', 'MarginSamplingSelection', 'EntropySelection', 'MinStdSelection']
        
        selection_functions_ssl = [SSLselection]
        selection_functions_ssl_str = ['SSLselection']
          
        
        models =     [   MLPModel  ,    KNNModel  ,  RfModel  ,  NBModel  ,   ExtModel ]
        models_str = [  'MLPModel' ,   'KNNModel' , 'RfModel' , 'NBModel' ,  'ExtModel' ]
        
        d, mytest = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1, L0, selection_functions_ssl, ssl_ratio, X_train_full, y_train_full, X_test, y_test, permutation, X_train, y_train)
        
        for i in mytest:
           print (i.shape)
        
        
        with open('al_ssl_nameofdataset_LR_' + str(LR) + '_ratio_1_' + str(ssl_ratio) + '_' + str(max_queried) + '_insta.pickle', 'wb') as f:
             pickle.dump(d,f)
        del d, mytest
    