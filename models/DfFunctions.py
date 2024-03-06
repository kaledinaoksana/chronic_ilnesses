class DfFunctions:

    # плотность матрицы
    @staticmethod
    def matrix_density(df):
        return df.sum().sum() / (df.shape[0] * df.shape[1]) * 100

    # поиск
    @staticmethod
    def find_row_by_user( dff, user, by="user_id"):
        return dff[(dff[by] == user)]

    @staticmethod
    def find_row_by_user_and_condition(dff, user, cond, by="user_id"):
        return dff[(dff[by] == user) & (dff["trackable_name"] == cond)]

    @staticmethod
    def find_rows_by_column_value(df, col, value):
        return df[(df[col] == value)]