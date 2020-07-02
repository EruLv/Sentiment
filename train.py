import numpy as np
import torch
from torch import nn
from bert_model import (
    BertBasedSentimentModel,
    BertBasedLSTMGRU,
    BertBasedSentimentModel2,
    BertBasedSentimentModel_last3)

from preparing_data import *
import argparse
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import BertConfig,RobertaConfig

#模型参数
PAD_IDX = 0
DROPOUT_PROB = 0.5
HIDDEN_DIM = 768
LSTM_HIDDEN_DIM = 256

#根据不同需求选择预训练模型结构，这个其实可以不用了，因为中文统统用bert
configs = {"bert":BertConfig,"roberta": RobertaConfig}


def train_epoch(model, train_data_loader,optimizer,schedualer, criterion, args):

    #训练一轮

    model.train()
    epoch_loss = 0

    for i, data in enumerate(train_data_loader):

        optimizer.zero_grad()

        inputs, labels, masks = data
        outputs = model(inputs, attention_mask = masks)

        loss = criterion(input=outputs, target=labels)

        loss.backward()
        optimizer.step()
        schedualer.step()

        if i % 10 == 0:
            print("Batch: {}, loss :{} ".format(i,loss.item()))

        epoch_loss  += loss.item()

    return epoch_loss / len(train_data_loader)


def evaluate(model,criterion, args):

    #使用验证集测试损失和准确率，返回验证集损失

    train_data = DataSet('valid',args)
    train_data_loader = train_data.get_data_loader()

    model.eval()
    epoch_loss = 0
    hit = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(train_data_loader):

            inputs, labels, masks = data
            outputs = model(inputs, attention_mask = masks)

            preds = torch.argmax(outputs, dim=1)  # 每一行的最大值下标

            loss = criterion(input=outputs, target=labels)

            hit += sum(labels == preds).item()
            total += len(labels)

            epoch_loss  += loss.item()

        print("valid acc: {}".format(hit/total))

    return epoch_loss / len(train_data_loader)


def train(args):

    train_data = DataSet('train',args)
    train_data_loader = train_data.get_data_loader(args.batch_size)
    n_epoch =args.n_epoch
    config = configs[args.model_name].from_pretrained(args.pretrained_model_name,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True
                                                           )
    criterion = nn.CrossEntropyLoss()

    if args.weight == 1:
        # 使用加权损失
        labels_count = get_count(args)
        print(labels_count)
        w = [labels_count[k] for k in sorted(labels_count.keys())]
        w = [x / sum(w) for x in w]  # 先求比例再求倒数
        w = [1 / x for x in w]  # 求倒数，数量越多，权重越小
        w = np.array([len(w) * x / sum(w) for x in w], dtype=np.float32)  # 归一化，使得weight sum = 3
        w = torch.from_numpy(w).to(args.device)
        print('weight:',w)
        # 损失函数采用加权交叉熵损失
        criterion = nn.CrossEntropyLoss(weight=w, reduction='sum')


    print("Training base model",args.model_id)
    if args.model_id == 1:
        model = BertBasedSentimentModel(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    elif args.model_id == 2:
        model = BertBasedSentimentModel2(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    elif args.model_id == 3:
        model = BertBasedSentimentModel_last3(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    else:
        model = BertBasedLSTMGRU(HIDDEN_DIM, LSTM_HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps= args.eps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps = len(train_data_loader) * n_epoch)

    min_valid_loss = 1000
    for epoch in range(n_epoch):
        #训练epoch轮
        train_loss = train_epoch(model,train_data_loader,optimizer,scheduler,criterion,args)
        valid_loss = evaluate(model,criterion,args)
        #比较之前的最好验证集损失，如果当前较好，覆盖之前模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model, 'models/best_valid_model_{}.pt'.format(args.model_id))
        #每轮骨骺保存模型，防止断电
        torch.save(model, 'models/last_model_{}.pt'.format(args.model_id))

        print("Epoch:{}, avg_train_loss: {}, avg_valid_loss: {}".format(epoch,train_loss,valid_loss))



def main():

    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=2,help = '每批数据的数量')
    parser.add_argument('--n_epoch',type=int,default= 5 ,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default= 0.001,help = '学习率')
    parser.add_argument('--eps',type=float,default= 1e-8,help = '近似0')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default= 2 ,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default= 6,help='分类类数')
    parser.add_argument('--warmup_steps',type=int,default= 0,help='Warm up')
    parser.add_argument('--weight',type=int,default= 0, help='是否使用加权交叉熵损失函数')
    #如果使用没用经过聚类的数据测试，那么将--data_path参数设置为'data/raw/',同时修改--num_labels为3
    parser.add_argument('--data_path',type=str,default='data/',help='数据集路径')
    parser.add_argument('--train_file',type=str,default='train.csv',help='训练集文件名')
    parser.add_argument('--valid_file',type=str,default='dev.csv',help='验证集文件名')
    parser.add_argument('--max_len',type=int,default=128,help='最大补全长度')

    parser.add_argument('--model_name',type=str,default='bert',help='预训练模型名')
    parser.add_argument('--model_id',type=int,default= 1,help='训练第几个模型')
    parser.add_argument('--pretrained_model_name',type=str,default='./models/chinese-roberta-wwm-ext',help='预训练模型地址')
    parser.add_argument('--tokenizer_name', type=str, default='./models/chinese-roberta-wwm-ext', help='分词器地址')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()