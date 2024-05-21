from sample import ml1m_sample_tfrecord
from core.model.YouTuBeDNN_ranking import RankingDNN
import numpy as np
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from feature import feature_processing


class SaveModelSignatures:

    def save_model(self,model,export_path):
        # 保存模型，默认签名  
        tf.saved_model.save(model, export_path)
  

    def save_model_signatures(self, model,  parsed_args, export_path):

        #tags input。先hash，再onehot（keras.layers.CategoryEncoding可指定最大范围）
        tags_dim = parsed_args["gender_hash_nums"] + parsed_args["age_hash_nums"] + parsed_args["occupation_hash_nums"] + parsed_args["seqlen_hash_nums"] 

        # 定义签名函数：预测（分类+回归多分类）  
        @tf.function(input_signature=[  
            tf.TensorSpec(shape=[None, parsed_args["userid_input_dim"]], dtype=tf.float32, name='user_id'),  
            tf.TensorSpec(shape=[None, parsed_args["movieid_input_dim"]], dtype=tf.float32, name='movie_id'), 
            tf.TensorSpec(shape=[None, parsed_args["rated_movies_lastN_input_dim"]], dtype=tf.float32, name='rated_movies_lastN'),             
            tf.TensorSpec(shape=[None, parsed_args["genres_input_dim"]], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, parsed_args["zip_id_input_dim"]], dtype=tf.float32, name='zip_id'),  
            tf.TensorSpec(shape=[None, tags_dim], dtype=tf.float32, name='tags') 
            
        ])

        def serve_predictions(user_id,movie_id,rated_movies_lastN,genres,zip_id ,tags):  
            return model([user_id,movie_id,rated_movies_lastN, genres, zip_id, tags])


        # 定义签名函数：提取特征  
        @tf.function(input_signature=[  
            tf.TensorSpec(shape=[None, parsed_args["userid_input_dim"]], dtype=tf.float32, name='user_id'),  
            tf.TensorSpec(shape=[None, parsed_args["movieid_input_dim"]], dtype=tf.float32, name='movie_id'), 
            tf.TensorSpec(shape=[None, parsed_args["rated_movies_lastN_input_dim"]], dtype=tf.float32, name='rated_movies_lastN'),             
            tf.TensorSpec(shape=[None, parsed_args["genres_input_dim"]], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, parsed_args["zip_id_input_dim"]], dtype=tf.float32, name='zip_id'),  
            tf.TensorSpec(shape=[None, tags_dim], dtype=tf.float32, name='tags') 

        ])        

        def extract_features(user_id,movie_id,rated_movies_lastN,genres,zip_id ,tags):  
            features = model.layers[-3].output  
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=features)  
            return intermediate_model([user_id,movie_id,rated_movies_lastN, genres, zip_id, tags])  


        # 创建签名字典  
        signatures = {  
            'serving_default': serve_predictions,   #或者对rating的分类和回归2个输出分布签名，可以加快推理
            'predict': serve_predictions,  
            'extract_features': extract_features  
        }  
        
        # 保存模型，包括多个签名  
        tf.saved_model.save(model, export_path, signatures=signatures)  
