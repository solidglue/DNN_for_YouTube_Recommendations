import sys
import pandas as pd
from tensorflow import keras
from keras import layers

class Features:

    def rating_sequences(self, data_df, seq_len_):

        
        # 构造评分序列特征
        #seq_data_df = data_df.groupby(['user_id'])['item_id'].agg(list).reset_index() #.agg(lambda x: '|'.join(x))  #.agg(list).reset_index() #
        #seq_data_df.columns = ["user_id","itemid_seq"]

        #按用户id、时间戳排序
        data_df_sorted = data_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])  
  
        # 分组取序列
        def rolling_itemids(series):  
            self.seq_len = seq_len_
            seq_items = ""
            items_list = list(series)
            if len(items_list) <= self.seq_len:
                self.seq_len = len(items_list) - 1
            for item in items_list[:self.seq_len - 1]:
                seq_items += str(item) + "|"
            seq_items += str(items_list[self.seq_len - 1])
            return seq_items

           # 分组取序列长度  
        def rolling_itemids_len(series):  
            self.seq_len = seq_len_
            items_list = list(series)
            if len(items_list) <= self.seq_len:
                self.seq_len = len(items_list) - 1
            return self.seq_len

        # 使用groupby和apply进行分组并应用滚动函数  
        seq_data_df1 = data_df_sorted.groupby(["user_id"])['item_id'].apply(rolling_itemids).reset_index(name='itemid_seq')  
        seq_data_df2 = data_df_sorted.groupby(["user_id"])['item_id'].apply(rolling_itemids_len).reset_index(name='itemid_seq_len')  
        seq_data_df = pd.merge(seq_data_df1, seq_data_df2)

        return seq_data_df

    def feature_hash(self,unhash_tensor, num_bins_, salt_='77'):
        #特征hash
        hash_layer = keras.layers.Hashing(num_bins=num_bins_, salt=salt_)
        hashed_tensor = hash_layer(unhash_tensor)

        return hashed_tensor
    
    def feature_onehot(self, vocabulary, unonehot_tensor):
        #类别one-hot
        lookup = layers.StringLookup(output_mode="one_hot")
        lookup.adapt(vocabulary)
        onhot_tensor = lookup(unonehot_tensor)

        return onhot_tensor

    