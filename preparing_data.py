import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,RobertaTokenizer
import numpy as np
import argparse
from sklearn.utils import shuffle
from tqdm import tqdm


tokenizers = {"bert":BertTokenizer,"roberta":RobertaTokenizer}

def get_data(args, data_type):
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
        self.args = args

        self.x = torch.from_numpy(x).to(torch.int64).to(self.args.device)
        self.y = torch.from_numpy(y).to(torch.int64).to(self.args.device)

        self.mask = torch.from_numpy(mask).to(torch.int64).to(self.args.device)
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index],  self.mask[index]

    def __len__(self):
        return self.len

class dataset():
    def __init__(self, data_type, args):

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

        max_len = 0
        for i in self.tok_texts:
            if len(i) > max_len:
                max_len = len(i)

        # print("max len: ", max_len)

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

        data_loader = DataLoader(dataset=self.dealDataset,
                                 batch_size=batch_size,
                                 shuffle = shuffle)
        return  data_loader

class text_encoder_decoder():
    def __init__(self,args):
        self.tokenizer = tokenizers[args.model_name].from_pretrained(args.tokenizer_name)

    def encode(self,texts):
        encode_texts = [self.tokenizer.encode(x,add_special_tokens=True)
                             for x in texts ]
        return encode_texts

    def decode(self, lst):
        #list[list[int]]
        decode_texts = [self.tokenizer.decode(x) for x in lst]
        return decode_texts

def split(fn):
    df = pd.read_csv(fn,sep = '\t')
    df = shuffle(df)

    df = df.dropna(axis=0)  #删掉空行

    df = df.reset_index(drop=True)
    df_train = df.loc[0:16000].reset_index(drop=True)
    df_test = df.loc[16000:18000].reset_index(drop=True)
    df_valid = df.loc[18000:].reset_index(drop=True)
    df_train.to_csv("data/train.csv",index=None)
    df_test.to_csv("data/test.csv",index=None)
    df_valid.to_csv("data/dev.csv",index=None)



if __name__ == '__main__':
    # split("data/data.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='每批数据的数量')
    parser.add_argument('--model_name',type=str,default='bert',help='预训练模型名')
    # parser.add_argument('--tokenizer_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='分词器')
    parser.add_argument('--tokenizer_name', type=str, default='models/chinese-roberta-wwm-ext', help='分词器')
    parser.add_argument('--data_path',type=str,default='data/',help='数据集路径')
    parser.add_argument('--train_file',type=str,default='train.csv',help='训练集文件名')
    parser.add_argument('--valid_file',type=str,default='dev.csv',help='验证集文件名')
    parser.add_argument('--test_file',type=str,default='test.csv',help='验证集文件名')
    parser.add_argument('--max_len',type=int,default=256,help='验证集文件名')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)

    args = parser.parse_args()

    dt = dataset("train",args)
    data_loader = dt.get_data_loader(args.batch_size)

    for index, data in enumerate(tqdm(data_loader)):
        inputs, labels, masks = data
        tqdm.write(str(index))

