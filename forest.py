import numpy as np
import pandas as pd
import h5py as h5

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

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
    
    team_file = 'data/team_forest.h5'
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
                
                estimator = RandomForestRegressor(n_estimators = 100)
                score = cross_val_score(estimator, xCol, y, cv = 5).mean()
                
                print('%s-r2: %.3f' % (col, score))
                if(score > maxScore):
                    print('----------New Best----------')
                    maxScore = score
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