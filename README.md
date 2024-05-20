# BiLSTM-CRF
## 中文命名实体识别
使用BiLSTM-CRF实现NER任务。  
能识别出句子中包含地名信息和机构名信息。  

* 详见 _《自然语言处理应用与实战》_ 电子工业出版社版 _BiLSTM-CRF的命名实体识别_
# 说明  
## 下载数据集  
* 阿里云盘链接  
BiLSTM-CRF
https://www.alipan.com/s/hx1UBEZz8rB
提取码: wt34  

数据集如下：  
```
--- BiLSTM-CRF
--- --- data
--- --- --- train.txt
--- --- --- test.txt
```
## 修改配置文件路径
将config.py文件中的路径修改为本地数据集存放路径
```
# 训练集路径
path_train = " "

# 测试集路径
path_test =  " "
```

# 使用
## 环境配置
Python == 3.10  
使用命令行配置环境
```
pip install -r requirements.txt
```
## 训练
数据集下载完成后，直接使用命令：
```
python train.py
```
