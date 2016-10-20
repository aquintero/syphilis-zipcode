import pandas as pd

def main():
    in_file = 'data/hills-comp.csv'
    out_file = 'data/syphilis-count.csv'
    
    df = pd.read_csv(in_file, sep = ',')
    zips = df['Zip'].unique()
    
    cases = dict()
    for zip in zips:
        cases[zip] = 0

    
    for zip in df['Zip'].values:
        cases[zip] = cases[zip] + 1
    
    cases_rows = []
    
    for zip in cases:
        cases_rows.append({'zip': zip, 'count': cases[zip]})
        
    cases_df = pd.DataFrame.from_records(cases_rows)
    cases_df.to_csv(out_file, sep = ',', index = False)
    
if __name__ == '__main__':
    main()