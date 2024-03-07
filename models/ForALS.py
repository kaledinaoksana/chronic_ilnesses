import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from models import TimeSeriesProcessor


proc = TimeSeriesProcessor.TimeSeriesDataProcessor()

class ForALS:

    # preparation to als
    @staticmethod
    def some_ids_preparation_to_als(user_item_matrix):

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values
        
        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))
        
        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))
        
        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return itemid_to_id, userid_to_id, id_to_itemid, id_to_userid
    
    # split to train, test, val percent
    @staticmethod
    def split_to_train_val_test_percent(df, percent_train = 0.7):

        data_sorted = df.sort_values(by=['checkin_date'])
        grouped_data = data_sorted.groupby(by='numeric_user_id')

        train_data = []
        test_data = []
        val_data = []

        for user_id, group_df in grouped_data:

            unique_dates = group_df['checkin_date'].unique()
            total_dates = len(unique_dates)
            train_count = int(total_dates * percent_train)
            test_count = int((total_dates - train_count) / 2)
        
            train_dates = unique_dates[:train_count]
            val_dates = unique_dates[train_count:train_count + test_count]
            test_dates = unique_dates[train_count + test_count:]

            train_df = group_df[group_df['checkin_date'].isin(train_dates)]
            test_df = group_df[group_df['checkin_date'].isin(test_dates)]
            val_df = group_df[group_df['checkin_date'].isin(val_dates)]

            train_data.append(train_df)
            test_data.append(test_df)
            val_data.append(val_df)

        train_df = pd.concat(train_data)
        test_df = pd.concat(test_data)
        val_df = pd.concat(val_data)

        return train_df, val_df, test_df
    
    # split to train, test - last day
    @staticmethod
    def split_to_train_test_last_day(df):
        """
        df_clean необходимо разделить на train и test: 
        test выбирает по последней дате записи пользователя 
        train: все до последней даты, 
        test: последняя дата
        """
        
        df_clean_sorted = df.sort_values(by=['numeric_user_id', 'checkin_date'])
        grouped_df = df_clean_sorted.groupby('numeric_user_id')

        train_data = []
        test_data = []

        for user_id, group_df in grouped_df:
            # Получаем последнюю дату для данного пользователя
            last_date = group_df['checkin_date'].max()
            # Выбираем все записи кроме последней даты для обучающего набора
            train_data.append(group_df[group_df['checkin_date'] != last_date])
            # Выбираем все записи на последнюю дату для тестового набора
            test_data.append(group_df[group_df['checkin_date'] == last_date])

        train_df = pd.concat(train_data)
        test_df = pd.concat(test_data)

        return train_df, test_df
    
    
    @staticmethod
    def figure_umap_embeddings(model_als, umap_emb, name):

        plt.figure(figsize=(10, 7))
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=10)  # Рассеиваем точки
        plt.title(name)  # Заголовок
        plt.xlabel('UMAP Component 1')  # Метка оси X
        plt.ylabel('UMAP Component 2')  # Метка оси Y
        plt.grid(True)  # Включаем сетку

        model_info = f"""
        Model: ALS, 
        Factors: {model_als.factors} 
        Regularization: {model_als.regularization}
        Iterations: {model_als.iterations}
        """
        
        # Добавление информации о модели в квадратике на графике
        if model_info:
            plt.text(0.77, 0.05, model_info, ha='left', va='bottom', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=1))
        
        # Определение save_path, чтобы избежать ошибки UnboundLocalError
        save_path = None

        if name == 'UMAP Visualization of User Embeddings':
            name = f"ALS_f{model_als.factors}_r{model_als.regularization}_i{model_als.iterations}"
            save_path = f'07_figures/UMAP_Users/{name}.png'

        # Сохранение графика в файл, если указан путь для сохранения
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show() 


    # cosine average_similarity
    @staticmethod
    def compute_cosine_similarity(cluster_embeddings):
        similarity_matrix = cosine_similarity(cluster_embeddings)
        average_similarity = similarity_matrix.mean()
        return average_similarity
    


class Clustering:

    # PRINT CLUSTER
    @staticmethod
    def print_clusters(X, labels, core_samples_mask,n_clusters_):
        size = 10
        s = 7
        plt.figure(figsize=(20, 10))

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            # noize
            if k == -1:
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
        
            xy = X[class_member_mask & core_samples_mask]
            if len(xy) > 0:
                centroid_x = xy[:, 0].mean()
                centroid_y = xy[:, 1].mean()
            else:
                centroid_x = 0
                centroid_y = 0
                
            xy = X[class_member_mask & core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], s=s, c=[col], label='Cluster %d' % k)

        
            # Добавляем метку кластера
            if k != -1:
                a=1
                centroid_x = xy[:, 0].mean()
                centroid_y = xy[:, 1].mean()
                xx=centroid_x
                yy=centroid_y+a
                plt.plot([centroid_x, xx], [centroid_y, yy], color='gray', linestyle='--')
                plt.text(xx, yy, str(k), fontsize=size, color='black',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))
            else:
                xy = X[class_member_mask & ~core_samples_mask]
                plt.scatter(xy[:, 0], xy[:, 1], s=s, c=[col])

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        pass


    #DBSCAN
    @staticmethod
    def print_dbscan(X, n_clusters):

        db = DBSCAN(eps=0.3, min_samples=n_clusters).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        cos_sim = {}

        for cluster_id in range(n_clusters_):
            cluster_points = X[labels == cluster_id]
            cos_sim[cluster_id] = ForALS.compute_cosine_similarity(cluster_points)

        print(core_samples_mask)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

        Clustering.print_clusters(X, labels, core_samples_mask, n_clusters_)
        cos_sim_df = pd.DataFrame(list(cos_sim.items()), columns=['Cluster #', 'Cosine Similarity'])

        return labels, cos_sim_df, n_clusters_


    #KMEANS
    @staticmethod
    def print_kmeans(X, n_clusters):

        kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init='auto')
        labels = kmeans.fit_predict(X)

        core_samples_mask = np.ones_like(labels, dtype=bool)

        cos_sim = {}

        for cluster_id in range(n_clusters):
            cluster_points = X[labels == cluster_id]
            cos_sim[cluster_id] = ForALS.compute_cosine_similarity(cluster_points)

        Clustering.print_clusters(X, labels, core_samples_mask, n_clusters)
        cos_sim_df = pd.DataFrame(list(cos_sim.items()), columns=['Cluster #', 'Cosine Similarity'])

        return labels, cos_sim_df, n_clusters
    

    # add clusters to df
    @staticmethod
    def create_df_with_cl(user_emb, labels, train_df):

        assert len(labels) == user_emb.shape[0], "Dimensions mismatch: cluster_labels and umap_emb"
        umap_emb_with_cluster_labels = np.hstack((user_emb, labels.reshape(-1, 1)))

        cluster_ids=proc.drop_columns_by_columnsnames(pd.DataFrame(umap_emb_with_cluster_labels),[0,1])
        cluster_ids['numeric_user_id'] = cluster_ids.index
        cluster_ids = cluster_ids.rename(columns={cluster_ids.columns[0]: 'cluster_id'})

        return pd.merge(train_df, cluster_ids, on="numeric_user_id", how="inner")

    
    # создаем словарь count_trname_by_cluster[cluster_id] с данными расчетов 
    # чаще всего встречающихся заболеваний у группы людей
    @staticmethod
    def create_count_trname_by_cluster_dict(n_clusters, train_df_with_clusters, result):
        count_trname_by_cluster = {}

        def find_rows_by_column_and_values(df, col, values):
            return df[df[col].isin(values)]

        def find_rows_by_column_value(df, col, value):
                return df[(df[col] == value)]

        for cluster_name_id in range(0, n_clusters):
            users_by_cluster = find_rows_by_column_value(train_df_with_clusters,'cluster_id',cluster_name_id)
            users = users_by_cluster.numeric_user_id.unique()
            df_user_results = find_rows_by_column_and_values(result,'numeric_user_id',users)
            # df_user_results = dfun.find_rows_by_column_value(result,'numeric_user_id',users)
            df_user_results.head(40)
            count_trname_by_cluster[cluster_name_id] = proc.calculate_counttable_by_columnname(users_by_cluster, 'numeric_trackable_name').sort_values(by="count", ascending=False)

        return count_trname_by_cluster


    # Словарь для хранения данных по кластерам и самых распространенных заболеваниях
    @staticmethod
    def create_usual_features_in_clusters(count_trname_by_cluster, threshold_by_item, threshold_by_main_group):
        clusters = {} 
        # порок слов 
        threshold_percentage = threshold_by_item
        threshold_all_percentage = threshold_by_main_group

        for cluster_id, df in count_trname_by_cluster.items():
            
            percent_of_total = 0
            percent = 0
            uslovie = 1
            per=1
            norm = False
            i=0
                
            total_count = df['count'].sum()
            sorted_df = df.sort_values(by='count', ascending=False)

            diseases = []

            while per > threshold_percentage:

                disease = sorted_df.iloc[i]['numeric_trackable_name']
                percent = sorted_df.iloc[i]['count'] / total_count
                percent_of_total = percent_of_total + percent

                if (percent > threshold_percentage):
                    norm=True
                    # print(disease, " - per = ", percent, ", total = ",percent_of_total)
                    diseases.append(disease)
                    i=i+1

                elif (norm==False) and (uslovie==1):
                    uslovie=0
                    j=0
                    percent_of_total=0

                    while percent_of_total < threshold_all_percentage:
                        disease = sorted_df.iloc[j]['numeric_trackable_name']
                        percent = sorted_df.iloc[j]['count'] / total_count
                        percent_of_total = percent_of_total + percent
                        j=j+1
                        if percent_of_total < threshold_all_percentage:
                            diseases.append(disease)
                        if j > 15:
                            break

                    i=i+1

                else:
                    break

            clusters[cluster_id] = diseases
        
        return clusters