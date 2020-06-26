# Sentiment
🐍暑期实训，进行中
### 😬模型介绍
基于roberta预训练模型，在其后面使用多个简单下游分类结构，包括:
* roberta \<CLS\> outputs -> fc softmax
* roberta ( \<CLS\> outputs + Pooled_outputs) -> fc softmax
* roberta (last 3 hidden states + Pooled_outputs) -> fc softmax
* roberta (Pooled_outputs + (outputs(\<CLS\> + all words represnetations) - > BiLSTM -> BiGRU ) ) -> fc softmax

优化器:sgd
 
### 😀已完成
* 数据迭代器
* 基础模型(4/5)
* Bagging模型融合
* 评测指标(准确率，精确率，召回率，F1)
* ~~Adam & AdamW~~实验效果不如SGD
* Warmup

### 😅待完成
* 中文数据应用(队友今天处理完数据了吗)
* 基础模型(1/5,凑个奇数)
* 给网站的模型接口

### 🚀当前指标(用来测试的英文数据集，不具有参考价值)
准确率: 0.9231191652937946, 平均精确率: 0.9230809594117306, 平均召回率: 0.9230809594117306, 平均F值: 0.9235640854609675





