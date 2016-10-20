import numpy as np
import pandas as pd
import h5py as h5

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

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
    
    team_file = 'data/team.h5'
    team_key = 'team'
    score_key = 'score'
    with h5.File(team_file) as hdf:
        if not team_key in hdf:
            hdf[team_key] = []
            hdf[score_key] = -10000
        team = hdf[team_key][:].tolist()
        while True:
            scores = dict()
            maxScore = hdf[score_key][...]
            print(maxScore)
            maxCol = ''
            for col in x.columns:
                if any(col == mem for mem in team):
                    continue
                print('testing %s ...' % col)
                team.append(col)
                xCol = x[team].values
                
                scaler = StandardScaler()
                scaler.fit(xCol)
                xCol = scaler.transform(xCol)
                
                estimator = SVR()
                C_space = np.logspace(-3, 3, 10)
                epsilon_space = np.logspace(-3, 3, 10)
    
                search_params = dict(
                    C = C_space,
                    epsilon = epsilon_space
                )
                
                search = GridSearchCV(estimator, param_grid = search_params, scoring = 'r2', cv = 5)
                search.fit(xCol, y)
                
                print('%s-r2: %.3f' % (col, search.best_score_))
                if(search.best_score_ > maxScore):
                    print('----------New Best----------')
                    maxScore = search.best_score_
                    maxCol = col
                
                team = [mem for mem in team if mem != col]
            if maxCol == '':
                break
                
            team.append(maxCol)
            del hdf[team_key]
            hdf[team_key] = team
            del hdf[score_key]
            hdf[score_key] = maxScore
            
            print('Added %s to team' % maxCol)
            print('Best Score: %.3f' % maxScore)
        
if __name__ == '__main__':
    main()