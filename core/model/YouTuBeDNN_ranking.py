import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RankingDNN:

    def build_model(self, parsed_args):
        #您可以使用函数式 API 通过几行代码构建此模型：
        #embedding input
        userid_input =  keras.Input(shape=(parsed_args["userid_input_dim"],), name="user_id")  #用户id
        itemid_input =  keras.Input(shape=(parsed_args["movieid_input_dim"],), name="movie_id") #内容id
        itemid_lastN_input =  keras.Input(shape=(parsed_args["rated_movies_lastN_input_dim"],), name="rated_movies_lastN") #评分序列 
        genres_input =  keras.Input(shape=(parsed_args["genres_input_dim"],), name="genres")  #类型
        zip_input =  keras.Input(shape=(parsed_args["zip_id_input_dim"],), name="zip_id")  #类型

        #tags input。先hash，再onehot（keras.layers.CategoryEncoding可指定最大范围）
        tags_dim = parsed_args["gender_hash_nums"] + parsed_args["age_hash_nums"] + parsed_args["occupation_hash_nums"] + parsed_args["seqlen_hash_nums"] 
        tags_input =  keras.Input(shape=(tags_dim,), name="tags")

        ## 单值类别特征embedding，例如userid
        userid_embeddings = layers.Embedding(parsed_args["userid_hash_nums"], parsed_args["embedding_out_dim"])(userid_input)
        userid_embeddings = layers.Flatten()(userid_embeddings)

        itemid_embeddings = layers.Embedding(parsed_args["itemid_hash_nums"], parsed_args["embedding_out_dim"])(itemid_input)
        itemid_embeddings = layers.Flatten()(itemid_embeddings)

        zipid_embeddings = layers.Embedding(parsed_args["zipid_hash_nums"], parsed_args["embedding_out_dim"])(zip_input)
        zipid_embeddings = layers.Flatten()(zipid_embeddings)

        ##单值类别特征embedding，例如评分序列
        lastN_multi_value_embedding = layers.Embedding(parsed_args["itemid_hash_nums"] , parsed_args["embedding_out_dim"])(itemid_lastN_input)
        itemid_lastN_mean_embeddings = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(lastN_multi_value_embedding)
        genres_multi_value_embedding = layers.Embedding(parsed_args["itemid_hash_nums"] , parsed_args["embedding_out_dim"])(genres_input) #此处复用序列长度空间，可单独设置
        genres_mean_embeddings = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(genres_multi_value_embedding)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([userid_embeddings, itemid_embeddings, itemid_lastN_mean_embeddings ,genres_mean_embeddings, zipid_embeddings, tags_input])
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)
        #x = layers.BatchNormalization()(x)  
        x = layers.Dense(64, activation='relu')(x)


        ##分类和回归2个数据去预测评分
        # Stick a logistic regression for priority prediction on top of the features
        rating_priority_pred = layers.Dense(1, name="score")(x)
        # Stick a department classifier on top of the features
        rating_class_pred = layers.Dense(parsed_args["num_ratings"], name="rating")(x)

        # Instantiate an end-to-end model predicting both priority and department
        model = keras.Model(
            inputs=[userid_input, itemid_input, itemid_lastN_input, genres_input, zip_input, tags_input],
            outputs=[rating_priority_pred, rating_class_pred],
        )

        return model
    