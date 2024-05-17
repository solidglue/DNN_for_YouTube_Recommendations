import sys
sys.path.append('d://the-recommender/DNN_for_YouTube_Recommendations')

import argparse
from core.trainner.trainner_cpu import TrainnerCPU
from sample import ml1m_sample_tfrecord
from core.io.model_save_and_signatures import SaveModelSignatures
from datasets.ml1m_dataset import Ml1mDataset 
import tensorflow as tf

def build_argparser(parser):

    #dateset配置
    parser.add_argument('--data_path', type=str, help='--export_path',
                        default='d://the-recommender/DNN_for_YouTube_Recommendations/datasets/ml-1m/') 
    parser.add_argument('--users_file', type=str, help='--users_file', default='users.dat') 
    parser.add_argument('--ratings_file', type=str, help='--ratings_file', default='ratings.dat') 
    parser.add_argument('--movies_file', type=str, help='--movies_file', default='movies.dat') 
    parser.add_argument('--sep', type=str, help='--sep', default='::') 
    parser.add_argument('--header', type=bool, help='--header', default=None) 



    #样本配置
    parser.add_argument('--sample_output_file', type=bool, help='--sample_output_path',
                         default='d://the-recommender/DNN_for_YouTube_Recommendations/sample/__TFRecord/train.tfrecord') 

    #模型保存配置
    parser.add_argument('--export_path', type=str, help='--export_path',
                        default='d://the-recommender/DNN_for_YouTube_Recommendations/SAVEDMODELS/savedmodel/default_signature_with_multiple_input_output')  
    parser.add_argument('--export_path_multi', type=str,  help='--export_path_multi',
                        default='d://the-recommender/DNN_for_YouTube_Recommendations/SAVEDMODELS/savedmodel/multi_signature_with_multiple_input_output') 
    #特征hash等配置 
    parser.add_argument('--userid_hash_nums', type=int,  help='--userid_hash_nums', default=10000)  
    parser.add_argument('--itemid_hash_nums', type=int,  help='--itemid_hash_nums', default=10000)  
    parser.add_argument('--zipid_hash_nums', type=int,  help='--zipid_hash_nums', default=10000)  
    parser.add_argument('--gender_hash_nums', type=int,  help='--gender_hash_nums', default=10)  
    parser.add_argument('--age_hash_nums', type=int, help='--age_hash_nums', default=100) 
    parser.add_argument('--occupation_hash_nums', type=int, help='--occupation_hash_nums', default=100) 
    parser.add_argument('--seqlen_hash_nums', type=int, help='--seqlen_hash_nums', default=100) 
    parser.add_argument('--hash_salt', type=int,  help='--hash_salt', default=1024) 
    parser.add_argument('--vocab_size', type=int,  help='--vocab_size', default=1000)  #标签序列词表大小
    parser.add_argument('--seq_len', type=int,  help='--seq_len', default=5)  #序列特征的长度，例如历史评分过的最近5个节目id序列

    #模型网络结构配置
    parser.add_argument('--userid_input_dim', type=int, help='--userid_input_dim', default=1) 
    parser.add_argument('--movieid_input_dim', type=int, help='--movieid_input_dim', default=1) 
    parser.add_argument('--rated_movies_lastN_input_dim', type=int, help='--rated_movies_lastN_input_dim', default=5) 
    parser.add_argument('--genres_input_dim', type=int, help='--genres_input_dim', default=6) 
    parser.add_argument('--zip_id_input_dim', type=int, help='--learning_rate', default=1) 
    parser.add_argument('--num_ratings', type=int, help='--num_ratings', default=6)
    parser.add_argument('--dense_size', type=int, help='--dense_size', default=256)
    parser.add_argument('--embedding_out_dim', type=int, help='--embedding_out_dim', default=64) 


    #模型训练配置
    parser.add_argument('--batch_size', type=int,  help='--batch_size', default=64)    
    parser.add_argument('--epochs', type=int,  help='--epochs', default=2) 
    parser.add_argument('--validation_split', type=float,  help='--validation_split', default=0.2)  
    parser.add_argument('--score_loss_weight', type=float,  help='--score_loss_weight', default=1) 
    parser.add_argument('--rating_loss_weight', type=float,  help='--rating_loss_weight', default=1) 
    parser.add_argument('--learning_rate', type=int, help='--learning_rate', default=1e-3) 



    return   parser

def parse_args(parser):
    args = parser.parse_args([])

    parsed_args = {}
    #解析dataset参数
    parsed_args["data_path"] =  args.data_path 
    parsed_args["users_file"] =  args.users_file 
    parsed_args["ratings_file"] =  args.ratings_file 
    parsed_args["movies_file"] =  args.movies_file 
    parsed_args["sep"] =  args.sep 
    parsed_args["header"] =  args.header 

    #解析样本参数
    parsed_args["sample_output_file"] =  args.sample_output_file 

    #解析模型保存参数
    parsed_args["export_path"] =  args.export_path 
    parsed_args["export_path_multi"] = args.export_path_multi

    #解析特征hash参数
    parsed_args["userid_hash_nums"] = args.userid_hash_nums
    parsed_args["itemid_hash_nums"] = args.itemid_hash_nums
    parsed_args["zipid_hash_nums"] = args.zipid_hash_nums
    parsed_args["gender_hash_nums"] = args.gender_hash_nums
    parsed_args["age_hash_nums"] = args.age_hash_nums
    parsed_args["occupation_hash_nums"] = args.occupation_hash_nums
    parsed_args["seqlen_hash_nums"] = args.seqlen_hash_nums
    parsed_args["hash_salt"] = args.hash_salt
    parsed_args["vocab_size"] = args.vocab_size
    parsed_args["seq_len"] = args.seq_len

    #解析模型网络结构参数
    parsed_args["userid_input_dim"] = args.userid_input_dim
    parsed_args["movieid_input_dim"] = args.movieid_input_dim
    parsed_args["rated_movies_lastN_input_dim"] = args.rated_movies_lastN_input_dim
    parsed_args["genres_input_dim"] = args.genres_input_dim
    parsed_args["zip_id_input_dim"] = args.zip_id_input_dim
    parsed_args["num_ratings"] = args.num_ratings    
    parsed_args["dense_size"] = args.dense_size    
    parsed_args["embedding_out_dim"] = args.embedding_out_dim

    #解析模型训练参数
    parsed_args["batch_size"] = args.batch_size
    parsed_args["epochs"] = args.epochs 
    parsed_args["validation_split"] = args.validation_split       
    parsed_args["score_loss_weight"] = args.score_loss_weight
    parsed_args["rating_loss_weight"] = args.rating_loss_weight 
    parsed_args["learning_rate"] = args.learning_rate

    return parsed_args


def main():

    ## 1.初始化参数
    trainner = TrainnerCPU()
    parser = trainner.build_argparser(argparse.ArgumentParser())
    parsed_args = parse_args(parser)

    ## 2.加载数据集
    data_loader = Ml1mDataset()
    data_X, data_y = data_loader.load_ml1m(parsed_args)

    ## 3.构造样本
    trainner = TrainnerCPU()
    signature = SaveModelSignatures()
    sample_num = 200000
    #从原始数据集构造样本,并转换为tfrecord格式
    train_dataset = ml1m_sample_tfrecord.Ml1mTFRecordSample().tfrecord_sample(data_X[:sample_num], data_y[:sample_num], True, parsed_args) 

    ## 4.训练模型
    model = trainner.train_loop(train_dataset, parsed_args)

    ## 5.保存模型 - 自定义签名
    SaveModelSignatures().save_model_signatures(model, parsed_args["export_path_multi"])

    ## 6.模型推理验证
    #特征处理, 构造输入tensor
    input_tensor_dict = trainner.build_input_tensor(train_dataset,parsed_args)

    #获取模型签名
    imported_with_signatures = tf.saved_model.load(parsed_args["export_path_multi"])
    serve_signature = imported_with_signatures.signatures['serving_default']  

    #推理验证,评分和分类
    predict_output = serve_signature(user_id=input_tensor_dict["hashed_userid_tensor"], 
                                    movie_id=input_tensor_dict["hashed_itemid_tensor"],
                                    rated_movies_lastN=input_tensor_dict["paded_itemid_lastN_tensor"],
                                    genres=input_tensor_dict["paded_genres_tensor"],
                                    zip_id=input_tensor_dict["hashed_zip_tensor"],
                                    tags=input_tensor_dict["tags_tensor"])  
    print(predict_output)


if __name__ == "__main__":  
    main()    