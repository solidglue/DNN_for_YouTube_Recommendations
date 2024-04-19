

import pandas as pd

class Ml1mDataset:

    def __init__(self):
        self.data_path = "d://the-recommender/DNN_for_YouTube_Recommendations/datasets/ml-1m/"
        self.users_file = self.data_path + 'users.dat'
        self.ratings_file = self.data_path + 'ratings.dat'
        self.movies_file = self.data_path + 'movies.dat'
        self.sep = '::'
        self.header = None

    def load_data(self, file_name, feature_columns):
        data_df = pd.read_csv(file_name, sep=self.sep, header=self.header, names=feature_columns, engine='python')
        return data_df
    
    def load_ml1m(self):

        #define columnsp
        users_columns = ['user_id','gender','age','occupation','zip']
        rating_columns = ['user_id','item_id','rating','timestamp']
        movies_columns = ['item_id','title','genres']

        #load ml1m
        users_df = self.load_data(self.users_file, users_columns)
        ratings_df = self.load_data(self.ratings_file, rating_columns)
        movies_df = self.load_data(self.movies_file, movies_columns)

        ml1m_df = pd.merge(pd.merge(ratings_df,movies_df),users_df)

        data_y = ml1m_df[["rating"]]
        data_x = ml1m_df.drop(["rating"],axis=1)

        return data_x, data_y
