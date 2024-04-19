import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        itemid_input =  keras.Input(shape=(None,), name="item_id") #内容id
        genres_input =  keras.Input(shape=(None,), name="genres")  #类型
        itemid_seq_input =  keras.Input(shape=(None,), name="itemid_seq") #评分序列 

        #tags input。指定维度，不好，拆分单个tensor不用指定
        tags_input =  keras.Input(shape=(None,173), name="tags")  #age/gender/occupation/zip-code/itemid_seq_len.  删掉timestamp和title特征。

        # Embed each word in the title into a 64-dimensional vector
        userid_features = layers.Embedding(self.userid_num_words, self.embeding_dim)(userid_input)
        itemid_features = layers.Embedding(self.itemid_num_words, self.embeding_dim)(itemid_input)
        genres_features = layers.Embedding(self.num_words, self.embeding_dim)(genres_input)
        itemid_seq_features = layers.Embedding(self.num_words, self.embeding_dim)(itemid_seq_input)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([userid_features, itemid_features,genres_features, itemid_seq_features, tags_input])
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)
        #x = layers.BatchNormalization()(x)  
        x = layers.Dense(64, activation='relu')(x)


        ##分类和回归2个数据去预测评分
        # Stick a logistic regression for priority prediction on top of the features
        rating_priority_pred = layers.Dense(1, name="priority")(x)
        # Stick a department classifier on top of the features
        rating_class_pred = layers.Dense(self.num_ratings, name="rating")(x)

        # Instantiate an end-to-end model predicting both priority and department
        model = keras.Model(
            inputs=[userid_input, itemid_input, genres_input,itemid_seq_input, tags_input],
            outputs=[rating_priority_pred, rating_class_pred],
        )

        return model
    