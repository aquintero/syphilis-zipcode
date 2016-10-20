import pandas as pd

def main():
    col_file = 'data/column-names.csv'
    in_file = 'data/hills-census.csv'
    out_file = 'data/hills-census_fixed.csv'
    
    col_names = pd.read_csv(col_file, sep = ',', header = None, names = ['id', 'name'])
    df = pd.read_csv(in_file, sep = ',')
    
    for index, row in col_names.iterrows():
        rename = dict()
        if row['id'] in df:
            rename[row['id']] = row['name']
        df = df.rename(columns = rename)
            
    df.to_csv(out_file, sep = ',', index = False)
    
if __name__ == '__main__':
    main()