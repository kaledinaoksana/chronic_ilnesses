
import pandas as pd
import inspect


class TimeSeriesDataProcessor:
    """
    A class for processing time series data.
    
    Attributes:
        None
    """
    
    @staticmethod
    def load_csv(filename, path):
        """
        Load CSV file.
        """
        
        file = path+filename+'.csv'
        return pd.read_csv(file, low_memory=False)
    
    @staticmethod
    def calculate_counttable_by_columnname(df, column_name):
        """
        Calculate a count table for a column.
        
        Args:
            df (DataFrame): The DataFrame to calculate the count table for.
            column_name (str): The name of the column to calculate the count table for.
            
        Returns:
            DataFrame: The count table.
        """
        counts = df[column_name].value_counts().reset_index()
        counts.columns = [column_name, 'count']
        counts = counts.sort_values(by='count')
        return counts
    
    @staticmethod
    def drop_columns_by_columnsnames(df, columns):
        """
        Drop columns from a DataFrame.
        
        Args:
            df (DataFrame): The DataFrame to drop columns from.
            columns (list): A list of column names to drop.
            
        Returns:
            DataFrame: The DataFrame with columns dropped.
        """
        if [col for col in columns if col in df.columns]:
            df = df.drop(columns=columns)
        return df
    
    @staticmethod
    def delete_rows_with_higher_count(df, df_count, min_count, col='trackable_name'):
        """
        Delete rows where the count is more than a specified value.
        
        Args:
            df (DataFrame): The DataFrame to delete rows from.
            df_count (DataFrame): The DataFrame containing the counts.
            min_count (int): The minimum count value.
            col (str): The column name to check the count for.
            
        Returns:
            DataFrame: The DataFrame with rows deleted where trackable_name > min_count.
        """
        merged_df = df.merge(df_count, on=col, how='inner')
        merged_df = merged_df.drop(merged_df.columns[0], axis=1)
        df_clean = merged_df[merged_df['count'] >= min_count]
        return df_clean
    

    @staticmethod
    def filter_by_min_records(df, num_of_records, by='user_id'):
        """
        Filter users(default) by minimum number of records.
    
        Args:
            df (DataFrame): The DataFrame to filter.
            num_of_records (int): The minimum number of records.
            by (str): The column name to group by.
            
        Returns:
            DataFrame: The filtered DataFrame.
        """
        user_counts = df.groupby(by).size()
        user_counts.name = 'count_of_records'
        users_records = user_counts[user_counts >= num_of_records]
        return pd.DataFrame(users_records).reset_index()
    
    @staticmethod
    def filter_by_min_number_of_dates(df, min_num_of_dates, by='user_id'):
        """
        Filter users(default) by minimum number of dates.
        
        Args:
            df (DataFrame): The DataFrame to filter.
            min_num_of_dates (int): The minimum number of dates.
            
        Returns:
            DataFrame: The filtered DataFrame.
        """
        user_dates_count = df.groupby(by)['checkin_date'].nunique()
        valid_users = user_dates_count[user_dates_count >= min_num_of_dates].index
        filtered_df = df[df['user_id'].isin(valid_users)]
        return filtered_df
    
    def create_mapping_matrix(self, df, column):
        """
        Create a mapping matrix between original and numeric values for a given column.

        Args:
            df (DataFrame): The DataFrame containing the column to map.
            column (str): The name of the column to map.

        Returns:
            DataFrame: A DataFrame with two columns - the original values and their corresponding numeric values.
            DataFrame: The original DataFrame with a new column containing the numeric values.
        """
        # Получение уникальных значений и их преобразование в числовые индексы
        unique_values = df[column].unique()
        numeric_values = list(range(len(unique_values)))
        
        # Создание словаря, сопоставляющего уникальные значения с числовыми индексами
        mapping_dict = dict(zip(unique_values, numeric_values))

        # Создание нового столбца с числовыми индексами
        new_column_name = 'numeric_' + column
        df[new_column_name] = df[column].map(mapping_dict)
        
        # Создание матрицы сопоставления
        mapping_matrix = pd.DataFrame({column: unique_values, new_column_name: numeric_values})

        return mapping_matrix, df

    # -----------------------
    @staticmethod
    def list_methods():
        """
        List all methods in the class.
        """
        for name, method in inspect.getmembers(TimeSeriesDataProcessor, inspect.isfunction):
            print("{name:35s} : {info}".format(name=name, info = inspect.getdoc(method).split('\n')[0]))