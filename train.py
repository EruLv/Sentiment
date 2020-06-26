import numpy as np
import torch
from torch import nn
from bert_model import (
    BertBasedSentimentModel,
    BertBasedLSTMGRU,
    BertBasedSentimentModel2,
    BertBasedSentimentModel_last3)

from preparing_data import dataset
import argparse
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import BertConfig,RobertaConfig

PAD_IDX = 0

DROPOUT_PROB = 0.5
HIDDEN_DIM = 768
LSTM_HIDDEN_DIM = 256

configs = {"bert":BertConfig,"roberta": RobertaConfig}


def train_epoch(model, train_data_loader,optimizer,schedualer, criterion, args):

    model.train()
    epoch_loss = 0

    for i, data in enumerate(train_data_loader):

        optimizer.zero_grad()

        inputs, labels, masks = data

        outputs = model(inputs, attention_mask = masks)
        #outputs: (batch_size, n_labels)

        loss = criterion(input=outputs, target=labels)

        loss.backward()

        optimizer.step()

        schedualer.step()

        if i % 10 == 0:
            print("Batch: {}, loss :{} ".format(i,loss.item()))

        epoch_loss  += loss.item()

    return epoch_loss / len(train_data_loader)


def evaluate(model,criterion, args):

    train_data = dataset('valid',args)
    train_data_loader = train_data.get_data_loader()

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, data in enumerate(train_data_loader):

            inputs, labels, masks = data
            outputs = model(inputs, attention_mask = masks)

            loss = criterion(input=outputs, target=labels)

            epoch_loss  += loss.item()

    return epoch_loss / len(train_data_loader)


def train(args):

    train_data = dataset('train',args)
    train_data_loader = train_data.get_data_loader()
    n_epoch =args.n_epoch
    config = configs[args.model_name].from_pretrained(args.pretrained_model_name,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True
                                                           )
    model = BertBasedSentimentModel(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    # model = BertBasedSentimentModel2(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    # model = BertBasedSentimentModel_last3(HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)
    # model = BertBasedLSTMGRU(HIDDEN_DIM, LSTM_HIDDEN_DIM, DROPOUT_PROB ,config, args).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps= args.eps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps = len(train_data_loader) * n_epoch)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        train_loss = train_epoch(model,train_data_loader,optimizer,scheduler,criterion,args)
        valid_loss = evaluate(model,criterion,args)
        print("Epoch:{}, avg_train_loss: {}, avg_valid_loss: {}".format(epoch,train_loss,valid_loss))

    torch.save(model, args.model_name +'_model_test.pt')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=8,help = '每批数据的数量')
    parser.add_argument('--n_epoch',type=int,default=2,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--eps',type=float,default=(1e-8),help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=2,help='分类类数')
    parser.add_argument('--warmup_steps',type=int,default=0,help='Warm up steps')

    parser.add_argument('--data_path',type=str,default='data/SST2/SST2/',help='数据集路径')
    parser.add_argument('--train_file',type=str,default='train.tsv',help='训练集文件名')
    parser.add_argument('--valid_file',type=str,default='dev.tsv',help='验证集文件名')

    parser.add_argument('--model_name',type=str,default='roberta',help='预训练模型名')
    # parser.add_argument('--pretrained_model_name',type=str,default='bert-base-uncased',help='预训练模型')
    # parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='分词器')
    parser.add_argument('--pretrained_model_name',type=str,default='roberta-base',help='预训练模型')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='分词器')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)

    args = parser.parse_args()

    print(args.pretrained_model_name)

    train(args)


if __name__ == '__main__':
    main()