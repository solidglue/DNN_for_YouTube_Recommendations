
from datasets.ml1m_dataset import Ml1mDataset 
from feature.feature_process import Features
import pandas as pd

#INFO: unit test & EDA


class Ml1mSample():

    def __init__(self) :
        self.data_X, self.data_y = Ml1mDataset().load_ml1m()
        self.seq_len = 5 #序列特征的长度，例如历史评分过的最近5个节目id序列

    def data_clean(self):
        data_df = pd.concat([self.data_X, self.data_y],axis=1)
        data_df_sorted = data_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])  

        #数据集里有大量数据是一个用户在同一秒对多个电影评分的数据，只保留一条
        df_no_duplicates = data_df_sorted.drop_duplicates(subset=['user_id', 'timestamp'],keep='last').reset_index(drop=True) 

        return df_no_duplicates



    def recall_sample():
        pass


    def ranking_sample(self):
        df_no_duplicates = self.data_clean()

        #构造id序列特征
        seq_df = Features().rating_sequences(df_no_duplicates, self.seq_len)  

        #样本拼接
        sample_df =pd.merge(df_no_duplicates, seq_df).fillna(0)
        data_y = sample_df[["rating"]]
        data_x = sample_df.drop(["rating"],axis=1)

        return data_x, data_y
