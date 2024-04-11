借助网上已有的代码写了这个，从单句输入开始，但是用我们的数据的时候存在问题。
![image](https://github.com/Fun4U321/Transformer_Classifier/assets/165357065/871cf807-5295-4f7d-8467-2c6b4e024c6c)
划分训练、测试集的时候数据维度不一样。排查时发现是因为存在空值，且个别label的数据量不够。
![image](https://github.com/Fun4U321/Transformer_Classifier/assets/165357065/6b0d688d-5c90-445f-8b95-f46f60ec4885)

    语义关系：label_mapping = {"支持": 1, "反对": 2, "补充": 3, "质疑": 4, "": 0}
    发言类型：speech_type_mapping = {"主意": 1, "论证": 2, "资料": 3, "疑问": 4, "": 0}
