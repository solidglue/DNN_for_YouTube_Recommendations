from sample import ml1m_sample
from sample import ml1m_sample_tfrecord
from core.model.YouTuBeDNN_ranking import RankingDNN
import numpy as np
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from feature import feature_process


class TrainnerCPU:

    def __init__(self):
        self.export_path = "d://the-recommender/DNN_for_YouTube_Recommendations/core/trainner/__SavedModel/signature_with_multiple_input_output"  
        self.export_path_multi_ = "d://the-recommender/DNN_for_YouTube_Recommendations/core/trainner/__SavedModel/multi_signature_with_multiple_input_output"  

    def sparse_dataset(self, train_dataset):
        #输入数据
        userid_list = []
        itemid_list = []
        genres_list = []
        itemid_seq_list = []
        gender_list = []
        age_list = []
        zip_list = []
        occupation_list = []
        itemid_seq_len_list = []
        rating_label_list = []
        rating_score_list = []

        for input_element, label_element in train_dataset:  #未batch
            userid_list.append(str(input_element["user_id"].numpy()))
            itemid_list.append(str(input_element["item_id"].numpy()))
            genres_list.append(str(input_element["genres"].numpy()))
            itemid_seq_list.append(str(input_element["itemid_seq"].numpy()))

            gender_list.append(input_element["gender"].numpy())
            age_list.append(input_element["age"].numpy())
            zip_list.append(input_element["zip"].numpy())
            occupation_list.append(input_element["occupation"].numpy())
            itemid_seq_len_list.append(input_element["itemid_seq_len"].numpy())

            rating_label_list.append(str(label_element["rating"].numpy()))
            rating_score_list.append(label_element["rating"].numpy())

        userid_tensor = tf.constant(userid_list)
        itemid_tensor = tf.constant(itemid_list)        
        genres_tensor = tf.constant(genres_list)
        itemid_seq_tensor = tf.constant(itemid_seq_list)

        gender_df = pd.DataFrame(gender_list,columns=["gender"], dtype=str)
        age_df = pd.DataFrame(age_list,columns=["age"], dtype=str) #np.float32) 会报错，先转str后边用到再转int
        zip_df = pd.DataFrame(zip_list,columns=["zip"], dtype=str)
        occupation_df = pd.DataFrame(occupation_list,columns=["occupation"], dtype=str)
        itemid_seq_len_df = pd.DataFrame(itemid_seq_len_list,columns=["itemid_seq_len"], dtype=str)
        tags_df = pd.concat([gender_df,age_df,zip_df,occupation_df,itemid_seq_len_df], axis=1)

        #TODO:先特征hash再onehot，固定特征列数，样本的增减都不会影响shape，输入给模型时更可控
        onehot_tags_df = pd.get_dummies(tags_df)

        #print(tags_df[:10])
        #print(tags_df.info())

        ##
        tags_tensor = tf.constant(onehot_tags_df)
        rating_score_tensor = tf.constant(rating_score_list,dtype=tf.float32)
        rating_label_tensor = tf.constant(rating_label_list)


        return userid_tensor,itemid_tensor,genres_tensor,itemid_seq_tensor,tags_tensor,rating_score_tensor,rating_label_tensor

    def build_input_tensor(self, train_dataset):

        #input
        userid_tensor,itemid_tensor,genres_tensor,itemid_seq_tensor,tags_tensor,rating_score_tensor,rating_label_tensor = self.sparse_dataset(train_dataset)

        ## 特征处理
        feature_processer = feature_process.Features()

        #类型onehot
        onehot_rating_label_tensor = feature_processer.feature_onehot(rating_label_tensor, rating_label_tensor)

        #特征hash
        hashed_userid_tensor = feature_processer.feature_hash(userid_tensor, num_bins_=10000, salt_=7777)
        hashed_itemid_tensor = feature_processer.feature_hash(itemid_tensor, num_bins_=10000, salt_=6666)

        #todo:序列里的节目id也要hash，如果不共享embedding，并非强制性。暂时把序列当成一个特征hash
        #todo:拆分成每个特征一个tensor，分别进行hash和onehot。暂时用dataframe onhot
        hashed_genres_tensor = feature_processer.feature_hash(genres_tensor, num_bins_=10000, salt_=7777)
        hashed_itemid_seq_tensor = feature_processer.feature_hash(itemid_seq_tensor, num_bins_=10000, salt_=6666)


        return hashed_userid_tensor,hashed_itemid_tensor,hashed_genres_tensor,hashed_itemid_seq_tensor,tags_tensor,rating_score_tensor,onehot_rating_label_tensor


    def train_loop(self,train_dataset):

        #model 
        model = RankingDNN().build_model()

        hashed_userid_tensor,hashed_itemid_tensor,hashed_genres_tensor,hashed_itemid_seq_tensor,tags_tensor,rating_score_tensor,onehot_rating_label_tensor  \
         = self.build_input_tensor(train_dataset)
        

        #编译此模型时，可以为每个输出分配不同的损失。甚至可以为每个损失分配不同的权重，以调整其对总训练损失的贡献。
        model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss=[
                keras.losses.MeanAbsoluteError(), #keras.losses.MeanSquaredError(),
                keras.losses.CategoricalCrossentropy(from_logits=True),
            ],
            loss_weights=[1.0, 1.0],
        )

        model.fit(
            {"user_id": hashed_userid_tensor, "item_id": hashed_itemid_tensor, "genres": hashed_genres_tensor,"itemid_seq":hashed_itemid_seq_tensor,"tags":tags_tensor},
            {"priority": rating_score_tensor, "rating": onehot_rating_label_tensor},
            epochs=1,
            batch_size=64,
            validation_split=0.2,
        )

        return model

    def save_model(self,model):
        # 保存模型，包括签名  
        tf.saved_model.save(model, self.export_path)
  

    def save_model_signatures(self, model):
        # 定义签名函数：预测  
        #@tf.function(input_signature=[tf.TensorSpec(shape=[None, num_features], dtype=tf.float32)])  
        @tf.function(input_signature=[  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='user_id'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='item_id'), 
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='itemid_seq'),             
            tf.TensorSpec(shape=[None, 173], dtype=tf.float32, name='tags')  
        ])
        # def serve_predictions(inputs):  
        #     return model(inputs)  
        def serve_predictions(user_id,item_id,genres,itemid_seq,tags):  
            return model([user_id,item_id,genres,itemid_seq,tags])


        # 定义签名函数：提取特征  
        #@tf.function(input_signature=[tf.TensorSpec(shape=[None, num_features], dtype=tf.float32)])  
        @tf.function(input_signature=[  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='user_id'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='item_id'), 
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='itemid_seq'),             
            tf.TensorSpec(shape=[None, 173], dtype=tf.float32, name='tags')  
        ])        
        # def extract_features(inputs):  
        #     features = model.layers[-2].output  
        #     intermediate_model = tf.keras.Model(inputs=model.input, outputs=features)  
        #     return intermediate_model(inputs)  
        def extract_features(user_id,item_id,genres,itemid_seq,tags):  
            features = model.layers[-3].output  
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=features)  
            return intermediate_model([user_id,item_id,genres,itemid_seq,tags])  


        # 创建签名字典  
        signatures = {  
            'serving_default': serve_predictions,   #或者对rating的分类和回归2个输出分布签名，可以加快推理
            'predict': serve_predictions,  
            'extract_features': extract_features  
        }  
        
        # 保存模型，包括多个签名  
        tf.saved_model.save(model, self.export_path_multi_, signatures=signatures)  


        