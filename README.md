# Deep Neural Networks for YouTube Recommendations


## Introduce
本项目只实现了论文中的排序模型，召回模型同理。项目核心代码都在".py"文件中，所有".ipynb"中的代码只用作演示和单元测试，独立于业务。项目做了一些扩展，例如不定长多值序列特征的embedding（例如用户评分电影序列），简单多目标学习（例如预测评分分数和评分分类）。  
This project only implements the ranking model in the paper for demonstration, and the recall model is the same.All core codes are in the ".py" files, and the ".ipynb" scripts are only used for Unit Test and demonstration purposes.Easy to join Attention in the future, this project has made some extensions on the ranking model in the paper, such as Multi values Embedding (such as "rated_movies_lastN" feature) and Simple Multi-Task learning.  

Paper：https://dl.acm.org/doi/abs/10.1145/2959100.2959190  
DataSet: [ml-1m(movie rating)](https://grouplens.org/datasets/movielens/1m/)  


## Notice
如果通过Github站内超链接打开Jupyter Notebook文件发生错误，可以点击根据 https://nbviewer.org 生成的“备用链接”间接访问对应文件。  
或者通过以下链接访问整个项目的站外备用链接，注意点击站外备用链接里的非Jupyter Notebook格式文件会跳转回到Github仓库内：  
●  [**DNN_for_YouTube_Recommendations**](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/tree/main/)  


## Model & Model Trainning
[**加载数据(Load Datasets)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/datasets/datasets_test.ipynb)       [~~*(备用链接)*~~](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/blob/main/datasets/datasets_test.ipynb)   
[**构造样本(Create Samples)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/sample/ml1m_sample_tfrecord_test.ipynb)       [~~*(备用链接)*~~](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/blob/main/sample/ml1m_sample_tfrecord_test.ipynb)  
[**模型定义(Modle Structure)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/model/YouTuBeDNN_ranking_test.ipynb)       [~~*(备用链接)*~~](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/model/YouTuBeDNN_ranking_test.ipynb)  
[**模型训练(Modle Trainning)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/trainner/trainner_cpu_test.ipynb)       [~~*(备用链接)*~~](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/trainner/multi_values_embedding_layer_test.ipynb)  
[**保存模型与签名(Save Modles)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/trainner/trainner_cpu_test.ipynb)       [~~*(备用链接)*~~]()  
[**模型推理(Modle Inference)**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/infer/infer_test.ipynb)       [~~*(备用链接)*~~](https://nbviewer.org/github/solidglue/DNN_for_YouTube_Recommendations/blob/main/core/trainner/multi_values_embedding_layer_test.ipynb)  
[**main.py**](https://github.com/solidglue/DNN_for_YouTube_Recommendations/blob/main/main.py)  


## Ranking model in the paper
Deep ranking network architecture  
![alt text](./res/ranking.png)  


## Ranking model in this project
Simple Multi-Task learning  ranking network architecture  
![alt text](./res/multi_input_and_output_model.png)  
