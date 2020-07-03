# Sentiment
🐍暑期实训，进行中
### 😎模型介绍
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
* 加权交叉熵损失方法
* 网站

### 😅待完成
* 测试加权交叉熵损失方法

### 🚀当前指标

| 模型| Accuracy | Macro_Percision| Macro_Recall | Macro_F1|
|---|---|---|---|---|
|Model 1|84.54|68.32|63.91|66.04|
|Model 2|83.47|65.74|65.84|65.74|
|Model 3|82.87|65.08|67.38|66.32|
|Model 4|83.64|66.18|65.95|66.18|
|Bagging|83.67|65.98|65.92|66.18|
|加权损失|72.31|62.27|69.93|65.88|
|加权损失+聚类|82.51|55.60|49.24|55.21|






