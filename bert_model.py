import numpy as np
import torch
from torch import nn

from transformers import BertModel,RobertaModel
from transformers import BertConfig, RobertaConfig
from transformers import BertPreTrainedModel

models = {"bert":BertModel,"roberta": RobertaModel}
configs = {"bert":BertConfig,"roberta": RobertaConfig}

class BertBasedSentimentModel(BertPreTrainedModel):

    def __init__(self, hidden_dim, dropout_prob , config, args):
        super().__init__(config)
        '''
        Descriptions: 模型1,利用bert <CLS> -> fc  

        Args:
            hidden_dim: bert的隐藏层维度，base版本对应768
            dropout_prob: drop out 概率
            config: bert :BertConfig类
            args: 训练参数，主要用到:
                args.model_name: "bert"or"roberta",如果chinese-roberta-wwm-ext模型，一律使用bert
                self.args.pretrained_model_name: 预训练模型的名字/地址
                args.num_labels: 预测的类别数
        '''
        self.args = args

        self.config = config

        self.bert = models[args.model_name].from_pretrained(self.args.pretrained_model_name,
                                                            config = config)
        self.num_labels = self.args.num_labels

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        '''
        Args:
            input_ids: 输入的向量，维度[batch size, max_len]
            attention_mask: mask矩阵，维度同input_ids,0表示padded index,1表示其他
        Returns:
            logits: 预测向量,维度[batch size, num_labels]
        '''
        x = self.bert(input_ids, attention_mask=attention_mask)
        # x[0]: (batch_size, src_len, hidden_size) : (8,67,768)

        #<CLS>位置对应的输出
        first_hidden_state = x[0][:,0,:]  # (batch_size, hidden_size) : (8, 768)

        x = self.dropout(first_hidden_state)

        logits = self.fc(x)

        return logits

class BertBasedLSTMGRU(BertPreTrainedModel):
    def __init__(self, hidden_dim,lstm_hidden_dim, dropout_prob , config,args):
        super().__init__(config)
        '''
        Descriptions: 模型4
            利用bert P_o + ( bert outputs -> BiLSTM -> BIGRU ->(last hidden outputs, avg pool, max_pool)) -> fc  
        Args:
            hidden_dim: bert的隐藏层维度，base版本对应768
            lstm_hidden_dim: BiRNN的隐藏层数量，当前为512
            dropout_prob: drop out 概率
            config: bert :BertConfig类
            args: 训练参数，主要用到:
                args.model_name: "bert"or"roberta",如果chinese-roberta-wwm-ext模型，一律使用bert
                self.args.pretrained_model_name: 预训练模型的名字/地址
                args.num_labels: 预测的类别数
        '''

        self.args = args

        self.num_labels = self.args.num_labels

        self.lstm_hidden_dim = lstm_hidden_dim

        self.hidden_dim = hidden_dim

        self.config = config
        self.bert = models[args.model_name].from_pretrained(self.args.pretrained_model_name,
                                                            config = config)

        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim,
                          num_layers=1, bidirectional=True, batch_first=True)

        self.gru = nn.GRU(lstm_hidden_dim*2,lstm_hidden_dim,
               num_layers=1, bidirectional=True, batch_first=True)


        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(lstm_hidden_dim * 6 + hidden_dim , self.num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        '''
        同模型1
        '''
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        bert_output = outputs[0]
        pooled_output = outputs[1]

        # print(bert_output.shape, pooled_output.shape)

        h_lstm, _ = self.lstm(bert_output)  # [bs, seq, output*dir]
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, 2 * self.lstm_hidden_dim)

        # print(h_lstm.shape,h_gru.shape,hh_gru.shape)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        # print(avg_pool.shape, max_pool.shape)

        # print(h_gru.shape, avg_pool.shape, hh_gru.shape, max_pool.shape, pooled_output.shape)
        h_conc_a = torch.cat(
            (avg_pool, hh_gru, max_pool, pooled_output), 1
        )
        # print(h_conc_a.shape)

        output = self.dropout(h_conc_a)
        logits = self.fc(output)

        return logits  # (loss), logits, (hidden_states), (attentions)

class BertBasedSentimentModel2(BertPreTrainedModel):

    def __init__(self, hidden_dim, dropout_prob , config,args):
        super().__init__(config)
        '''
        Descriptions: 模型2
            利用bert P_o + <CLS> -> fc  
        Args:
            同模型1
        '''
        self.args = args

        self.num_labels = self.args.num_labels

        self.hidden_dim = hidden_dim
        self.config = config

        self.bert = models[args.model_name].from_pretrained(self.args.pretrained_model_name,
                                                            config = config)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_dim * 2, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        '''
        同模型1
        '''
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        bert_output = outputs[0]
        pooled_output = outputs[1]  #((batch_size, hidden_size) : (8, 768)
        first_hidden_state = bert_output[:, 0, :]  # (batch_size, hidden_size) : (8, 768)

        x = torch.cat((first_hidden_state, pooled_output), 1)

        x = self.dropout(x)

        logits = self.fc(x)

        return logits

class BertBasedSentimentModel_last3(BertPreTrainedModel):

    def __init__(self, hidden_dim, dropout_prob , config, args):
        super().__init__(config)
        '''
        Descriptions: 模型2
            利用bert P_o + (最后最后三个隐藏层<CLS>对应的输出(第0个位置)) -> fc  
        Args:
            同模型1
        '''
        self.args = args

        self.num_labels = self.args.num_labels

        self.hidden_dim = hidden_dim

        self.bert = models[args.model_name].from_pretrained(self.args.pretrained_model_name,
                                                            config = config)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_dim * 4, self.num_labels)


    def forward(self, input_ids=None, attention_mask=None):
        '''
        同模型1
        '''
        bert_output_a, pooled_output_a, hidden_output_a = self.bert(input_ids, attention_mask=attention_mask)
        last_cat = torch.cat(
            (pooled_output_a, hidden_output_a[-1][:, 0], hidden_output_a[-2][:, 0],
             hidden_output_a[-3][:, 0]),
            1,
        )
        logits = self.fc(last_cat)

        return logits


