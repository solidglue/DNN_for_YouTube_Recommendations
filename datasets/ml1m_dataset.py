

import pandas as pd

class Ml1mDataset:

    def read_data(self,file_name, feature_columns, parsed_args):
        data_df = pd.read_csv(file_name, sep=parsed_args["sep"], header=parsed_args["header"], names=feature_columns, engine='python')
        return data_df
    
    def load_ml1m(self, parsed_args):
        #define columnsp
        users_columns = ['user_id','gender','age','occupation','zip']
        rating_columns = ['user_id','item_id','rating','timestamp']
        movies_columns = ['item_id','title','genres']

        users_file_name = parsed_args["data_path"] + parsed_args["users_file"]
        ratings_file_name = parsed_args["data_path"] + parsed_args["ratings_file"]
        movies_file_name = parsed_args["data_path"] + parsed_args["movies_file"]

        #load ml1m
        users_df = self.read_data(users_file_name, users_columns, parsed_args)
        ratings_df = self.read_data(ratings_file_name, rating_columns, parsed_args)
        movies_df = self.read_data(movies_file_name, movies_columns, parsed_args)
        ml1m_df = pd.merge(pd.merge(ratings_df,movies_df),users_df)
        data_y = ml1m_df[["rating"]]
        data_x = ml1m_df.drop(["rating"],axis=1)

        return data_x, data_y
