# Deep Neural Networks for YouTube Recommendations

Updating...

## Introduce
This project only implements the ranking model in the paper for demonstration, and the recall model is the same.  
All core codes are in the ".py" files, and the ".ipynb" scripts are only used for Unit Test and demonstration purposes.  
Easy to join Attention in the future, this project has made some extensions on the ranking model in the paper, such as Multi values Embedding (such as "rated_movies_lastN" feature) and Simple Multi-Task learning.  

Paper：https://dl.acm.org/doi/abs/10.1145/2959100.2959190  
DataSet: [ml-1m(movie rating)](https://grouplens.org/datasets/movielens/1m/)  


## ranking model in the paper
Deep ranking network architecture  
![alt text](./res/ranking.png)  


## ranking model in this project
Simple Multi-Task learning  ranking network architecture 
![alt text](./res/multi_input_and_output_model.png)  
