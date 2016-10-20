import numpy as np
import pandas as pd
import h5py as h5

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

def main():
    team_file = 'data/team.h5'
    team_key = 'team'
    score_key = 'score'
    with h5.File(team_file, 'r') as hdf:
        team = hdf[team_key][:]
        print(team)
        score = hdf[score_key][...]
        print score
        
        x_file = 'data/hills-census_fixed.csv'
        y_file = 'data/syphilis-count.csv'
        
        x_df = pd.read_csv(x_file, sep = ',')
        y_df = pd.read_csv(y_file, sep = ',')
        
        x = x_df[team].values
        x_df = x_df.merge(y_df, how = 'left', left_on = 'ZCTA5CE10', right_on = 'zip')
        x_df['count'].fillna(0, inplace = True)
        y = x_df['count'].values
        
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        
        estimator = SVR()
        C_space = np.logspace(-5, 5, 100)
        epsilon_space = np.logspace(-5, 5, 100)

        search_params = dict(
            C = C_space,
            epsilon = epsilon_space
        )
        
        search = GridSearchCV(estimator, param_grid = search_params, scoring = 'r2', cv = 5)
        search.fit(x, y)
        
        print(search.best_score_)
        print(search.best_params_)
        
if __name__ == '__main__':
    main()