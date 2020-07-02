import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,RobertaTokenizer
import numpy as np
import argparse
from sklearn.utils import shuffle
from collections import Counter


tokenizers = {"bert":BertTokenizer,"roberta":RobertaTokenizer}

def get_data(args, data_type):
    '''
    Descriptions:
        根据需求加载csv文件
    Args:
        args:
            args.data_path: 所有数据集所在的文件夹名，如"data/"
        data_type: train or valid or test
    Returns:
        返回对应文件的Dataframe
    '''
    path = args.data_path
    if data_type == "train":
        path = path + args.train_file
    elif data_type == "valid":
        path = path + args.valid_file
    else:
        path = path + args.test_file

    return pd.read_csv(path, sep=',')

class DealDataset(Dataset):
    def __init__(self, x, y, mask = None, args = None):
        '''
        Descriptions:
            继承自pytorch的Dataset类，生成数据迭代器
        Args:
            x: 输入 [N, max len], N时所有样本数量
            y: 标签
            mask: mask 矩阵 [N, max len]
        '''
        self.args = args

        self.x = torch.from_numpy(x).to(torch.int64).to(self.args.device)
        self.y = torch.from_numpy(y).to(torch.int64).to(self.args.device)

        self.mask = torch.from_numpy(mask).to(torch.int64).to(self.args.device)
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index],  self.mask[index]

    def __len__(self):
        return self.len

class DataSet():
    def __init__(self, data_type, args):
        '''
        Descriptions:
            自定义数据类,有加载数据、分词、文本转index、补全、生成输入向量和mask向量的功能
        Args:
            data_type: 'train' or 'test' or 'valid
            args:
                args.max_len：最大补全长度
                args.device：cuda or cpu
        '''
        self.args = args

        self.padding_max = args.max_len

        self.data = get_data(args, data_type)
        self.tokenizer = tokenizers[args.model_name].from_pretrained(args.tokenizer_name)

        self.device = args.device

        self.tok_texts = self.data['1'].apply((lambda x:
                                         self.tokenizer.encode(x,add_special_tokens=True))).values

        self.labels = self.data['0'].values

        self.padded, self.mask= self._padding()

        self.dealDataset = DealDataset(self.padded,self.labels, self.mask, args)

    def _padding(self):
        '''
        Descriptions:
            将每个样本补全到最大长度，超出的部分忽略，生成mask向量
        Returns:
            padded:补全后的向量矩阵
            attention_mask: mask矩阵
        '''
        max_len = 0
        for i in self.tok_texts:
            if len(i) > max_len:
                max_len = len(i)

        if max_len > self.padding_max:
            max_len = self.padding_max

        padded = []

        for i in self.tok_texts:
            if(len(i) > max_len):
                padded.append(i[:max_len -1 ] +[i[-1]])
            else:
                padded.append(i +  [self.tokenizer.pad_token_id] * (max_len - len(i)))

        padded = np.array(padded)

        attention_mask = np.where(padded != self.tokenizer.pad_token_id, 1, 0)    #mask 机制
        return padded, attention_mask

    def get_data_loader(self,batch_size = 8, shuffle=True):
        '''
        Descriptions:
            获取数据迭代器
        Args:
            batch_size: batch size
            shuffle: 是否打乱数据
        Returns:
            数据迭代器
        '''
        data_loader = DataLoader(dataset=self.dealDataset,
                                 batch_size=batch_size,
                                 shuffle = shuffle)
        return  data_loader

class TextEncoderDecoder():
    def __init__(self,args):
        '''
        Descriptions:
            辅助功能函数，用于给网站的接口，功能是根据文本转成向量和根据向量还原文本
        '''
        self.tokenizer = tokenizers[args.model_name].from_pretrained(args.tokenizer_name)

    def encode(self,texts):
        '''
        Descriptions:
            将文本分词并转换成向量
        Args:
            tests: list[str]
        Returns:
            encode_texts: list[list[index]]
        '''
        encode_texts = [self.tokenizer.encode(x,add_special_tokens=True)
                             for x in texts ]
        return encode_texts

    def decode(self, lst):
        '''
        Descriptions:
            将文本向量转换为文本
        Args:
            lst: list[list[index]]
        Returns:
            decode_texts：list[str]
        '''
        decode_texts = [self.tokenizer.decode(x) for x in lst]
        return decode_texts

def split(fn):
    '''
    Descriptions:
        一个辅助寒素，目的是将data.csv划分为训练集、验证集、测试集
    Args:
        fn: 文件路径
    Returns:
        生成train.csv，dev.csv,test.csv
    '''
    df = pd.read_csv(fn,sep = '\t')
    df = shuffle(df)

    df = df.dropna(axis=0)  #删掉空行

    df = df.reset_index(drop=True)
    df_train = df.loc[0:16000].reset_index(drop=True)
    df_test = df.loc[16000:18000].reset_index(drop=True)
    df_valid = df.loc[18000:].reset_index(drop=True)
    df_train.to_csv("data/raw/train.csv",index=None)
    df_test.to_csv("data/raw/test.csv",index=None)
    df_valid.to_csv("data/raw/dev.csv",index=None)

def get_count(args):
    '''
    Descriptions:
        获取训练集中每个类被的样本数量，用于加权损失
    Returns:
        disc: {1:1233,2:12153,.....}
    '''

    df = get_data(args,'train')
    counter = Counter(df['0'])
    return dict(counter)




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='每批数据的数量')
    parser.add_argument('--model_name',type=str,default='bert',help='预训练模型名')
    # parser.add_argument('--tokenizer_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='分词器')
    parser.add_argument('--tokenizer_name', type=str, default='models/chinese-roberta-wwm-ext', help='分词器')
    parser.add_argument('--data_path',type=str,default='data/raw/',help='数据集路径')
    parser.add_argument('--train_file',type=str,default='train.csv',help='训练集文件名')
    parser.add_argument('--valid_file',type=str,default='dev.csv',help='验证集文件名')
    parser.add_argument('--test_file',type=str,default='test.csv',help='验证集文件名')
    parser.add_argument('--max_len',type=int,default=256,help='验证集文件名')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)

    args = parser.parse_args()

    get_count(args)

    # dt = DataSet("train",args)
    # data_loader = dt.get_data_loader(args.batch_size)


