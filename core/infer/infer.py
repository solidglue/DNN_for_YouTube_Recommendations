import tensorflow as tf

class Inference:

    def get_infer(self, model_path):
        #查看模型签名 - 多签名（结果预测和取特征）
        imported_model_with_signatures = tf.saved_model.load(model_path)
        #print(list(imported_model_with_signatures.signatures.keys())) #['serving_default', 'predict', 'extract_features']

        ## 推理验证签名
        # 使用签名进行推理  
        #serve_signature_infer = imported_model_with_signatures.signatures['serving_default']  
        #extract_features_signature_infer = imported_model_with_signatures.signatures['extract_features']  

        return imported_model_with_signatures.signatures

