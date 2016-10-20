import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSCanonical
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR, LinearSVR

def main():
    x_file = 'data/hills-census_fixed.csv'
    y_file = 'data/syphilis-count.csv'
    
    ignore = ['FID', 'ZCTA5CE10', 'GEOID10', 'ALAND10', 'AWATER10', 'INTPTLAT10', 'INTPTLON10']
    
    x_df = pd.read_csv(x_file, sep = ',')
    y_df = pd.read_csv(y_file, sep = ',')
    
    x = x_df.drop(ignore, axis = 1)
    x_df = x_df.merge(y_df, how = 'left', left_on = 'ZCTA5CE10', right_on = 'zip')
    x_df['count'].fillna(0, inplace = True)
    y = x_df['count'].values
    
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    filter = RFECV(LinearSVR(), cv = 5)
    filter.fit(x, y)
    x = x[:, filter.ranking_[filter.ranking_ < 50]]
    print(x.shape)
    pls = PLSCanonical(n_components = 3)
    pls.fit(x, y)
    x, y = pls.transform(x, y)
    
    estimator = SVR()
    C_space = np.logspace(-3, 3, 10)
    epsilon_space = np.logspace(-3, 3, 10)
    
    search_params = dict(
        C = C_space,
        epsilon = epsilon_space
    )
    
    search = GridSearchCV(estimator, param_grid = search_params, scoring = 'r2', cv = 5)
    search.fit(x, y)
    
    print(search.best_score_)
    print(search.best_params_)
    print(search.best_estimator_.score(x, y))
    
if __name__ == '__main__':
    main()