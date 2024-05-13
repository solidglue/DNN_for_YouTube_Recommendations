from sample import ml1m_sample
from sample import ml1m_sample_tfrecord
from core.model.YouTuBeDNN_ranking import RankingDNN
import numpy as np
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from feature import feature_process

#TODO:所有输入参数在此处通过argparse配置
#TODO:tags里的特征先hash再合并，确保输入维度固定（或者reshape？）
#TODO:保存模型和签名放在io包里
#TODO:解析样本放在sample包里 - build_sample, sparse_sample

class TrainnerCPU:

    def __init__(self):
        self.export_path = "d://the-recommender/DNN_for_YouTube_Recommendations/core/trainner/__SavedModel/signature_with_multiple_input_output"  
        self.export_path_multi_ = "d://the-recommender/DNN_for_YouTube_Recommendations/core/trainner/__SavedModel/multi_signature_with_multiple_input_output"  

    def sparse_dataset(self, train_dataset):
        #输入数据
        userid_list, itemid_list, genres_list,  itemid_seq_list, gender_list, age_list, zip_list, occupation_list, itemid_seq_len_list, rating_label_list, rating_score_list = [[],[],[],[],[],[],[],[],[],[],[]]
        parsed_tensor_dict = {}

        for input_element, label_element in train_dataset:  #未batch
            userid_list.append(str(input_element["user_id"].numpy()))
            itemid_list.append(str(input_element["item_id"].numpy()))
            genres_list.append(str(input_element["genres"].numpy()))
            itemid_seq_list.append(str(input_element["itemid_lastN"].numpy()))

            gender_list.append(input_element["gender"].numpy())
            age_list.append(input_element["age"].numpy())
            zip_list.append(str(input_element["zip"].numpy()))
            occupation_list.append(input_element["occupation"].numpy())
            itemid_seq_len_list.append(input_element["itemid_lastN_len"].numpy())

            rating_label_list.append(str(label_element["rating"].numpy()))
            rating_score_list.append(label_element["rating"].numpy())

        ## 为了控制输入维度，修改成先hash后onehot，为了使用tf的hash函数，确保hash函数的一致性，不在此处hash和onehot
        # gender_df = pd.DataFrame(gender_list,columns=["gender"], dtype=str)
        # age_df = pd.DataFrame(age_list,columns=["age"], dtype=str) #np.float32) 会报错，先转str后边用到再转int
        # zip_df = pd.DataFrame(zip_list,columns=["zip"], dtype=str)
        # occupation_df = pd.DataFrame(occupation_list,columns=["occupation"], dtype=str)
        # itemid_seq_len_df = pd.DataFrame(itemid_seq_len_list,columns=["itemid_seq_len"], dtype=str)
        # tags_df = pd.concat([gender_df,age_df,zip_df,occupation_df,itemid_seq_len_df], axis=1)

        # #TODO:先特征hash再onehot，固定特征列数，样本的增减都不会影响shape，输入给模型时更可控
        # onehot_tags_df = pd.get_dummies(tags_df)

        # #print(tags_df[:10])
        # #print(tags_df.info())

        # ##
        # tags_tensor = tf.constant(onehot_tags_df)

        parsed_tensor_dict["userid_tensor"] = tf.constant(userid_list)
        parsed_tensor_dict["itemid_tensor"] =  tf.constant(itemid_list)     
        parsed_tensor_dict["genres_tensor"] = tf.constant(genres_list)
        parsed_tensor_dict["itemid_lastN_tensor"] = tf.constant(itemid_seq_list)
        parsed_tensor_dict["gender_tensor"] = tf.constant(gender_list)
        parsed_tensor_dict["age_tensor"] =  tf.constant(age_list)     
        parsed_tensor_dict["zip_tensor"] = tf.constant(zip_list)
        parsed_tensor_dict["occupation_tensor"] =  tf.constant(occupation_list)
        parsed_tensor_dict["itemid_lastN_len_tensor"] = tf.constant(itemid_seq_len_list)
        parsed_tensor_dict["rating_score_tensor"] =  tf.constant(rating_score_list,dtype=tf.float32)
        parsed_tensor_dict["rating_label_tensor"] =  tf.constant(rating_label_list)

        return parsed_tensor_dict

    def build_input_tensor(self, train_dataset):
        input_tensor_dict = {}

        #input
        parsed_tensor_dict = self.sparse_dataset(train_dataset)

        ## 特征处理
        feature_processer = feature_process.Features()

        #label, 类型label onehot, 数值label保持
        input_tensor_dict["onehot_rating_label_tensor"] = feature_processer.feature_onehot(parsed_tensor_dict["rating_label_tensor"], parsed_tensor_dict["rating_label_tensor"])
        input_tensor_dict["rating_score_tensor"] = parsed_tensor_dict["rating_score_tensor"]
        
        #特征hash - 单值，for embedding
        input_tensor_dict["hashed_userid_tensor"] = feature_processer.feature_hash(parsed_tensor_dict["userid_tensor"], num_bins_=10000, salt_=7777)
        input_tensor_dict["hashed_itemid_tensor"] = feature_processer.feature_hash(parsed_tensor_dict["itemid_tensor"], num_bins_=10000, salt_=6666)
        input_tensor_dict["hashed_zip_tensor"] = feature_processer.feature_hash(parsed_tensor_dict["zip_tensor"], num_bins_=10000, salt_=6666)

        #tag tensor合并
        #tag特征先hash后onehot主要是为了限制未知新值
        hashed_gender_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["gender_tensor"], num_bins_=10, salt_=6666).numpy(), dtype=int)
        hashed_age_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["age_tensor"], num_bins_=100, salt_=6666).numpy(), dtype=int)
        hashed_occupation_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["occupation_tensor"], num_bins_=100, salt_=6666).numpy(), dtype=int)
        hashed_itemid_seq_len_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["itemid_lastN_len_tensor"], num_bins_=100, salt_=6666).numpy(), dtype=int)
        tags_df = pd.concat([hashed_gender_df, hashed_age_df, hashed_occupation_df, hashed_itemid_seq_len_df], axis=1)

        #TODO:先特征hash再onehot，固定特征列数，样本的增减都不会影响shape，输入给模型时更可控
        #onehot_tags_df = pd.get_dummies(tags_df)
        #input_tensor_dict["tags_tensor"] = tf.constant(onehot_tags_df)

        onehot_layer_10 = keras.layers.CategoryEncoding(num_tokens=10, output_mode="one_hot")
        onehot_layer_100 = keras.layers.CategoryEncoding(num_tokens=100, output_mode="one_hot")
        
        onehot_gender_tensor = onehot_layer_10(tf.constant(hashed_gender_df))
        onehot_age_tensor = onehot_layer_100(tf.constant(hashed_age_df))
        onehot_occupation_tensor = onehot_layer_100(tf.constant(hashed_occupation_df))
        onehot_itemid_seq_len_tensor = onehot_layer_100(tf.constant(hashed_itemid_seq_len_df))

        input_tensor_dict["tags_tensor"] = tf.concat([onehot_gender_tensor, onehot_age_tensor, onehot_occupation_tensor ,onehot_itemid_seq_len_tensor],axis=1)

        #input_tensor_dict["tags_tensor"] = tf.constant(tags_df)

        #特征hash - 多值
        #todo:序列里的节目id也要hash，如果不共享embedding，并非强制性。暂时把序列当成一个特征hash
        #todo:拆分成每个特征一个tensor，分别进行hash和onehot。暂时用dataframe onhot
        input_tensor_dict["hashed_genres_tensor"] = feature_processer.feature_hash(parsed_tensor_dict["genres_tensor"], num_bins_=10000, salt_=7777)
        input_tensor_dict["hashed_itemid_lastN_tensor"] = feature_processer.feature_hash(parsed_tensor_dict["itemid_lastN_tensor"], num_bins_=10000, salt_=6666)

        return input_tensor_dict


    def train_loop(self,train_dataset):

        #model 
        model = RankingDNN().build_model()

        input_tensor_dict = self.build_input_tensor(train_dataset)
        

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
            {"user_id": input_tensor_dict["hashed_userid_tensor"], "movie_id": input_tensor_dict["hashed_itemid_tensor"], "genres": input_tensor_dict["hashed_genres_tensor"],
                "rated_movies_lastN":input_tensor_dict["hashed_itemid_lastN_tensor"],"zip_id":input_tensor_dict["hashed_zip_tensor"],"tags":input_tensor_dict["tags_tensor"]},
            {"score": input_tensor_dict["rating_score_tensor"], "rating": input_tensor_dict["onehot_rating_label_tensor"]},
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
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='movie_id'), 
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='rated_movies_lastN'),             
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='zip_id'),  
            tf.TensorSpec(shape=[None, 310], dtype=tf.float32, name='tags')  
        ])
        # def serve_predictions(inputs):  
        #     return model(inputs)  
        def serve_predictions(user_id,movie_id,rated_movies_lastN,genres,zip_id ,tags):  
            return model([user_id,movie_id,genres,rated_movies_lastN,zip_id, tags])


        # 定义签名函数：提取特征  
        #@tf.function(input_signature=[tf.TensorSpec(shape=[None, num_features], dtype=tf.float32)])  
        @tf.function(input_signature=[  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='user_id'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='movie_id'), 
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='rated_movies_lastN'),             
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='genres'),  
            tf.TensorSpec(shape=[None, ], dtype=tf.float32, name='zip_id'),  
            tf.TensorSpec(shape=[None, 310], dtype=tf.float32, name='tags')  
        ])        
        # def extract_features(inputs):  
        #     features = model.layers[-2].output  
        #     intermediate_model = tf.keras.Model(inputs=model.input, outputs=features)  
        #     return intermediate_model(inputs)  
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
        tf.saved_model.save(model, self.export_path_multi_, signatures=signatures)  


        