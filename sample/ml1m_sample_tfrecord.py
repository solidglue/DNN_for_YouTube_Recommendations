#import os  
import tensorflow as tf  
import numpy as np  
import pandas as pd
from feature.feature_processing import Features

class Ml1mTFRecordSample:

    # 定义一个函数来将样本写入 TFRecord  
    def write_tfrecord(self,sample_x, sample_y, output_path):  
        with tf.io.TFRecordWriter(output_path) as writer:  
            for user_id,item_id,timestamp,title,genres, gender,age,occupation,zip_,itemid_seq,itemid_seq_len,rating   \
                in zip(sample_x["user_id"],sample_x["item_id"],sample_x["timestamp"],sample_x["title"],sample_x["genres"] \
                       ,  sample_x["gender"],sample_x["age"],sample_x["occupation"],sample_x["zip"],sample_x["itemid_seq"],sample_x["itemid_seq_len"],sample_y["rating"]):  
               
                # 创建一个 tf.train.Example 对象  
                example = tf.train.Example(features=tf.train.Features(feature={  
                    'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),  
                    'item_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id])),  
                    'timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),  
                    'title': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title.encode('utf-8') ])), 
                    'genres': tf.train.Feature(bytes_list=tf.train.BytesList(value=[genres.encode('utf-8') ])),  
                    'gender': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gender.encode('utf-8') ])), 
                    'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),  
                    'occupation': tf.train.Feature(int64_list=tf.train.Int64List(value=[occupation])), 
                    'zip': tf.train.Feature(bytes_list=tf.train.BytesList(value=[zip_.encode('utf-8') ])),
                    'itemid_lastN': tf.train.Feature(bytes_list=tf.train.BytesList(value=[itemid_seq.encode('utf-8') ])),
                    'itemid_lastN_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[itemid_seq_len])),
                    'rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[rating]))
                    }))
                
                # 将 tf.train.Example 序列化为字符串并写入 TFRecord 文件 
                writer.write(example.SerializeToString())  


    # 定义解析函数，用于从 TFRecord 中提取图像和标签  
    def parse_example(self, example_proto):  
        features = {  
            'user_id': tf.io.FixedLenFeature([], tf.int64),   #要指定shape，否则默认是1个元素
            'item_id': tf.io.FixedLenFeature([], tf.int64),  
            'timestamp': tf.io.FixedLenFeature([], tf.int64),   
            'title': tf.io.FixedLenFeature([], tf.string),  
            'genres': tf.io.FixedLenFeature([], tf.string),
            'gender': tf.io.FixedLenFeature([], tf.string),  
            'age': tf.io.FixedLenFeature([], tf.int64),              
            'zip': tf.io.FixedLenFeature([], tf.string),  
            'itemid_lastN': tf.io.FixedLenFeature([], tf.string),              
            'itemid_lastN_len': tf.io.FixedLenFeature([], tf.int64),  
            'occupation': tf.io.FixedLenFeature([], tf.int64),              
            'rating': tf.io.FixedLenFeature([], tf.int64),         
        }  
        parsed_features = tf.io.parse_single_example(example_proto, features)  
        feature_columns = ["user_id","item_id","timestamp","title","genres", "gender","age","occupation","zip","itemid_lastN","itemid_lastN_len" ]
        label_clomuns = ["rating"]
        x_dict = {key: parsed_features[key] for key in feature_columns if key in parsed_features} 
        y_dict = {key: parsed_features[key] for key in label_clomuns if key in parsed_features} 

        return x_dict, y_dict
    

    def tfrecord_sample(self, data_X = pd.DataFrame(), data_y = pd.DataFrame(), is_write=False, parsed_args={}):

        feature_processor = Features()
        data_df = pd.concat([data_X, data_y],axis=1)
        data_df_sorted = data_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])  

        #数据集里有大量数据是一个用户在同一秒对多个电影评分的数据，只保留一条
        df_no_duplicates = data_df_sorted.drop_duplicates(subset=['user_id', 'timestamp'],keep='last').reset_index(drop=True) 
        #构造id序列特征
        seq_df = feature_processor.rating_sequences(df_no_duplicates, int(parsed_args["seq_len"]))  
        #样本拼接
        sample_df =pd.merge(df_no_duplicates, seq_df).fillna(0)
        sample_y = sample_df[["rating"]]
        sample_x = sample_df.drop(["rating"],axis=1)

        if is_write:
            self.write_tfrecord(sample_x, sample_y, parsed_args["sample_output_file"])  #可以从原始数据构造样本，也可以从tfrecord读取样本

        # 加载数据集  
        # 假设我们有一个 TFRecord 文件 'train.tfrecord'  
        dataset = tf.data.TFRecordDataset(parsed_args["sample_output_file"])  
        dataset = dataset.map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)  

        return dataset
