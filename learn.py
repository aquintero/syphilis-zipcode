import os
import pickle

import numpy as np
import pandas as pd
import h5py as h5

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSSVD
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR

class AdaBoostSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid
    
    def fit(self, X, y):
        self.team = []
        maxScore = -100000
        while True:
            scores = dict()
            maxCol = -1
            for col in range(X.shape[1]):
                if col in self.team:
                    continue
                print('testing feature %d ...' % col)
                self.team.append(col)
                
                tX = X[:, self.team]
                scaler = StandardScaler()
                scaler.fit(tX)
                tX = scaler.transform(tX)
                
                cv = KFold(n_splits = 5, shuffle = True, random_state = np.random.randint(1000))
                search = GridSearchCV(self.estimator, param_grid = self.param_grid, scoring = 'r2', cv = cv)
                search.fit(tX, y)
                
                print('r2: %.3f' % search.best_score_)
                if(search.best_score_ > maxScore):
                    print('----------New Best----------')
                    maxScore = search.best_score_
                    maxCol = col
                
                self.team.pop()
            if maxCol == -1:
                break
                
            self.team.append(maxCol)
            
            print('Added %s to team' % maxCol)
            print('Best Score: %.3f' % maxScore)
        
    def transform(self, X):
        return X[:, self.team]
        
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, param_grid, selection, reduction):
        self.pipeline = []
        self.pipeline.append(StandardScaler())
        if(reduction == 'pca'):
            self.pipeline.append(PCA(n_components = 10))
        if(selection == 'kbest'):
            self.pipeline.append(SelectKBest(k = 5))
        if(selection == 'ada'):
            self.pipeline.append(AdaBoostSelector(estimator, param_grid))
        
    def fit(self, X, y):
        for i in range(len(self.pipeline)):
            self.pipeline[i].fit(X, y)
            X = self.pipeline[i].transform(X)
            
    def transform(self, X):
        for i in range(len(self.pipeline)):
            X = self.pipeline[i].transform(X)
        return X
    
        
def main():
    np.random.seed(0)

    x_file = 'data/hills-census_fixed.csv'
    y_file = 'data/syphilis-count.csv'
    
    train_dir = 'data/train'
    test_dir = 'data/test'
    preprocesser_dir = 'data/preprocesser'
    if(not os.path.isdir(train_dir)):
        os.makedirs(train_dir)
    if(not os.path.isdir(test_dir)):
        os.makedirs(test_dir)
    if(not os.path.isdir(preprocesser_dir)):
        os.makedirs(preprocesser_dir)

    ignore = ['FID', 'ZCTA5CE10', 'GEOID10', 'ALAND10', 'AWATER10', 'INTPTLAT10', 'INTPTLON10']
    
    x_df = pd.read_csv(x_file, sep = ',')
    y_df = pd.read_csv(y_file, sep = ',')
    
    x = x_df.drop(ignore, axis = 1).values   
    x_df = x_df.merge(y_df, how = 'left', left_on = 'ZCTA5CE10', right_on = 'zip')
    x_df['count'].fillna(0, inplace = True)
    y = x_df['count'].values

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = np.random.randint(1000))
    
    estimators = [
        ('lasso', LassoCV(selection = 'random', random_state = np.random.randint(1000)), dict()),
        ('ridge', RidgeCV(), dict()),
        ('elastic', ElasticNetCV(selection = 'random', random_state = np.random.randint(1000)), dict()),
        ('forest', RandomForestRegressor(random_state = np.random.randint(1000)), dict(n_estimators = np.arange(50, 300, 50))),
        ('svr', SVR(), dict(C = np.logspace(-3, 3, 10), epsilon = np.logspace(-3, 3, 10), gamma = np.logspace(-3, 3, 10))),
        ('linearsvr', LinearSVR(), dict(C = np.logspace(-3, 3, 10)))
    ]    
    
    selections = [None, 'kbest', 'ada']
    reductions = [None, 'pca']
    
    for i in range(len(selections)):
        for j in range(len(reductions)):
            for k in range(len(estimators)):
                model = estimators[k][0]
                estimator = estimators[k][1]
                param_grid = estimators[k][2]
                selection = selections[i]
                reduction = reductions[j]
                
                model_name = model
                
                if selection is not None:
                    model_name = '%s_%s' % (model_name, selection)
                if reduction is not None:
                    model_name = '%s_%s' % (model_name, reduction)
                model_name = '%s' % model_name
                
                train_results_file = '%s/%s.csv' % (train_dir, model_name)
                test_results_file = '%s/%s.csv' % (test_dir, model_name) 
                preprocessor_file = '%s/%s.pkl' % (preprocesser_dir, model_name)
                
                if(os.path.isfile(test_results_file)):
                    continue
                
                pre = Preprocessor(estimator, param_grid, selection, reduction)
                pre.fit(train_x, train_y)
                with open(preprocessor_file, 'wb') as preprocessor_handle:
                    pickle.dump(pre, preprocessor_handle, pickle.HIGHEST_PROTOCOL)
                
                pretrain_x = pre.transform(train_x)
                pretest_x = pre.transform(test_x)
                
                cv = KFold(n_splits = 5, shuffle = True, random_state = np.random.randint(1000))
                search = GridSearchCV(estimator, param_grid = param_grid, cv = cv)
                search.fit(pretrain_x, train_y)
                
                train_results = pd.DataFrame(search.cv_results_)
                train_results.to_csv(train_results_file, sep = ',', index = False)
                
                estimator.set_params(**search.best_params_)
                estimator.fit(pretrain_x, train_y)
                score = estimator.score(pretest_x, test_y)
                
                test_results = pd.DataFrame(dict(r2 = [score]))
                test_results.to_csv(test_results_file, sep = ',', index = False)
                

if __name__ == '__main__':
    main()