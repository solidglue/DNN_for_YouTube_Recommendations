import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape


#TODO:输入维度固定
#TODO:多值embedding


class RankingDNN:

    def __init__(self):
        self.userid_num_words = 100000  # Size of vocabulary obtained when preprocessing text data
        self.itemid_num_words = 100000  
        self.num_words = 10000 
        self.embeding_dim = 64
        self.num_ratings = 6  # Number of departments for predictions

    def build_model(self):
        #您可以使用函数式 API 通过几行代码构建此模型：
        #embedding input
        userid_input =  keras.Input(shape=(None,), name="user_id")  #用户id
        itemid_input =  keras.Input(shape=(None,), name="movie_id") #内容id
        itemid_lastN_input =  keras.Input(shape=(None,), name="rated_movies_lastN") #评分序列 
        genres_input =  keras.Input(shape=(None,), name="genres")  #类型
        zip_input =  keras.Input(shape=(None,), name="zip_id")  #类型

        #tags input。先hash，再onehot（keras.layers.CategoryEncoding可指定最大范围）
        tags_input =  keras.Input(shape=(None,310), name="tags")  #age/gender/occupation/zip-code/itemid_seq_len.  删掉timestamp和title特征。


        # Embed each word in the title into a 64-dimensional vector
        userid_features = layers.Embedding(self.userid_num_words, self.embeding_dim)(userid_input)
        itemid_features = layers.Embedding(self.itemid_num_words, self.embeding_dim)(itemid_input)
        itemid_lastN_features = layers.Embedding(self.num_words, self.embeding_dim)(itemid_lastN_input)
        genres_features = layers.Embedding(self.num_words, self.embeding_dim)(genres_input)
        zipid_features = layers.Embedding(self.num_words, self.embeding_dim)(zip_input)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([userid_features, itemid_features, itemid_lastN_features ,genres_features, zipid_features, tags_input])
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)
        #x = layers.BatchNormalization()(x)  
        x = layers.Dense(64, activation='relu')(x)


        ##分类和回归2个数据去预测评分
        # Stick a logistic regression for priority prediction on top of the features
        rating_priority_pred = layers.Dense(1, name="score")(x)
        # Stick a department classifier on top of the features
        rating_class_pred = layers.Dense(self.num_ratings, name="rating")(x)

        # Instantiate an end-to-end model predicting both priority and department
        model = keras.Model(
            inputs=[userid_input, itemid_input, itemid_lastN_input, genres_input, zip_input, tags_input],
            outputs=[rating_priority_pred, rating_class_pred],
        )

        return model
    