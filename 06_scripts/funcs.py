
import numpy as np
import pandas as pd

class MyFunctions:

    def load_csv(filename):
        # file_loc = 'files/'+filename+'.csv'
        return pd.read_csv(filename)

    # create count table
    def calculate_count_table(df,column_name):
        counts = df[column_name].value_counts().reset_index()
        counts.columns = [column_name, 'count']
        
        counts = counts.sort_values(by='count')
        return counts

    def calculate_count_table_with_null(df,column_name):
        counts = df[column_name].value_counts().reset_index()
        counts.columns = [column_name, 'count']
        counts = counts.sort_values(by='count')
        
        null_count = df[column_name].isnull().sum()
        null_row = pd.DataFrame({column_name: ['null'], 'count': [null_count]})
        counts = pd.concat([counts, null_row], ignore_index=True)
        
        return counts