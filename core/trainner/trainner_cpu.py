
from core.model.YouTuBeDNN_ranking import RankingDNN
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from feature import feature_processing

class TrainnerCPU:

    ##解析tfrecord样本
    def sparse_dataset(self, train_dataset):
        userid_list, itemid_list, genres_list,  itemid_seq_list, gender_list, age_list, zip_list, occupation_list, itemid_seq_len_list, rating_label_list, rating_score_list = [[],[],[],[],[],[],[],[],[],[],[]]
        parsed_tensor_dict = {}

        for input_element, label_element in train_dataset:  
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

        #特征转tensor
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

    ## 构造输入tensor
    def build_input_tensor(self, train_dataset, parsed_args):
        input_tensor_dict = {}
        feature_processer = feature_processing.Features()

        ##解析数据，train_dataset是tfrecord格式的样本
        parsed_tensor_dict = self.sparse_dataset(train_dataset)

        ## 特征处理
        #label -  类型label onehot, 数值label保持不变
        input_tensor_dict["onehot_rating_label_tensor"] = feature_processer.feature_onehot(parsed_tensor_dict["rating_label_tensor"], parsed_tensor_dict["rating_label_tensor"])
        input_tensor_dict["rating_score_tensor"] = parsed_tensor_dict["rating_score_tensor"]
        
        #非序列单值类别特征hash - 后续 embedding
        input_tensor_dict["hashed_userid_tensor"] = tf.cast(feature_processer.feature_hash(parsed_tensor_dict["userid_tensor"], num_bins_=parsed_args["userid_hash_nums"], salt_=parsed_args["hash_salt"]), dtype=tf.float32)
        input_tensor_dict["hashed_itemid_tensor"] = tf.cast(feature_processer.feature_hash(parsed_tensor_dict["itemid_tensor"], num_bins_=parsed_args["itemid_hash_nums"], salt_=parsed_args["hash_salt"]), dtype=tf.float32)
        input_tensor_dict["hashed_zip_tensor"] = tf.cast(feature_processer.feature_hash(parsed_tensor_dict["zip_tensor"], num_bins_=parsed_args["zipid_hash_nums"], salt_=parsed_args["hash_salt"]), dtype=tf.float32)

        #tag特征hash，tensor合并，不embedding。 tag特征先hash后onehot主要是为了限制空间避免未知新值影响空间大小
        hashed_gender_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["gender_tensor"], num_bins_=parsed_args["gender_hash_nums"], salt_=parsed_args["hash_salt"]).numpy(), dtype=int)
        hashed_age_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["age_tensor"], num_bins_=parsed_args["age_hash_nums"], salt_=parsed_args["hash_salt"]).numpy(), dtype=int)
        hashed_occupation_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["occupation_tensor"], num_bins_=parsed_args["occupation_hash_nums"], salt_=parsed_args["hash_salt"]).numpy(), dtype=int)
        hashed_itemid_seq_len_df = pd.DataFrame(feature_processer.feature_hash(parsed_tensor_dict["itemid_lastN_len_tensor"], num_bins_=parsed_args["seqlen_hash_nums"], salt_=parsed_args["hash_salt"]).numpy(), dtype=int)

        onehot_layer_10 = keras.layers.CategoryEncoding(num_tokens=10, output_mode="one_hot")
        onehot_layer_100 = keras.layers.CategoryEncoding(num_tokens=100, output_mode="one_hot")
        onehot_gender_tensor = onehot_layer_10(tf.constant(hashed_gender_df))
        onehot_age_tensor = onehot_layer_100(tf.constant(hashed_age_df))
        onehot_occupation_tensor = onehot_layer_100(tf.constant(hashed_occupation_df))
        onehot_itemid_seq_len_tensor = onehot_layer_100(tf.constant(hashed_itemid_seq_len_df))
        input_tensor_dict["tags_tensor"] = tf.concat([onehot_gender_tensor, onehot_age_tensor, onehot_occupation_tensor ,onehot_itemid_seq_len_tensor],axis=1)

        ## 不定长序列特征，pad补齐到固定长度
        input_tensor_dict["paded_itemid_lastN_tensor"] = tf.constant(feature_processer.input_pad_sequences(parsed_tensor_dict["itemid_lastN_tensor"]), dtype=tf.float32)
        input_tensor_dict["paded_genres_tensor"]  = tf.constant(feature_processer.input_pad_sequences(parsed_tensor_dict["genres_tensor"]), dtype=tf.float32)
        
        return input_tensor_dict

    ## 模型训练
    def train_loop(self,train_dataset, parsed_args):

        #创建model 
        model = RankingDNN().build_model(parsed_args)

        #构造输入
        input_tensor_dict = self.build_input_tensor(train_dataset, parsed_args)
        
        #多目标学习。 编译此模型时，可以为每个输出分配不同的损失。甚至可以为每个损失分配不同的权重，以调整其对总训练损失的贡献。
        model.compile(
            optimizer=keras.optimizers.RMSprop(parsed_args["learning_rate"]),
            loss=[
                keras.losses.MeanAbsoluteError(), #keras.losses.MeanSquaredError(),
                keras.losses.CategoricalCrossentropy(from_logits=True),
            ],
            loss_weights=[parsed_args["score_loss_weight"], parsed_args["rating_loss_weight"]],
        )


        #模型训练
        model.fit(
            {"user_id": input_tensor_dict["hashed_userid_tensor"], "movie_id": input_tensor_dict["hashed_itemid_tensor"], 
            "rated_movies_lastN":input_tensor_dict["paded_itemid_lastN_tensor"], "genres": input_tensor_dict["paded_genres_tensor"],
            "zip_id":input_tensor_dict["hashed_zip_tensor"],"tags":input_tensor_dict["tags_tensor"]},
            {"score": input_tensor_dict["rating_score_tensor"], "rating": input_tensor_dict["onehot_rating_label_tensor"]},
            epochs = parsed_args["epochs"] ,
            batch_size = parsed_args["batch_size"],
            validation_split = parsed_args["validation_split"],
        )

        return model
