import sys
import pandas as pd
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Features:

    def rating_sequences(self, data_df, seq_len_):
        ## 构造评分序列特征
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

    def input_pad_sequences(self, input_sequences_tensor):
        # 示例输入数据  input_texts = ['a|b|c', 'b|c', 'a|b|c|d', 'e|f|g|h|i']  
        input_sequences_list = input_sequences_tensor.numpy().tolist()
        input_sequences_list = [s.decode('utf-8') for s in input_sequences_list]

        # 使用Tokenizer将文本转换为序列  
        tokenizer = Tokenizer(filters='|')  # 使用'|'作为分隔符  
        tokenizer.fit_on_texts(input_sequences_list)  
        sequences = tokenizer.texts_to_sequences(input_sequences_list)  
    
        # 计算最大序列长度  
        max_sequence_length = max(len(seq) for seq in sequences)  
        # 使用pad_sequences填充序列到相同长度  
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')  

        return padded_sequences
