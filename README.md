# Sentiment
🐍暑期实训，进行中
### 😬模型介绍
基于roberta预训练模型，在其后面使用多个简单下游分类结构，包括:
* roberta \<CLS\> outputs -> fc softmax
* roberta ( \<CLS\> outputs + Pooled_outputs) -> fc softmax
* roberta (last 3 hidden states + Pooled_outputs) -> fc softmax
* roberta (Pooled_outputs + (outputs(\<CLS\> + all words represnetations) - > BiLSTM -> BiGRU ) ) -> fc softmax

优化器:sgd

Max Lenghth: 128
 
### 😀已完成
* 数据清洗
* 样本数量平衡(聚类)
* 数据迭代器
* 基础模型×4
* Bagging模型融合
* 评测指标(准确率，精确率，召回率，F1)
* ~~Adam & AdamW~~ 实验效果不如SGD
* Warmup

### 😅待完成
* 与网站对接

### 🚀当前指标(用来测试的英文数据集，不具有参考价值)

| 模型| Accuracy | Macro_Percision| Macro_Recall | Macro_F1|
|Model 1|84.54|68.32|63.91|66.04|
|Model 2|83.47|65.74|65.84|65.74|
|Model 3|82.84|66.12|67.53|66.12|
|Model 4|83.64|66.18|65.95|66.18|
|Bagging|83.67|~~66.92~~|65.92|66.18|






