import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,RobertaTokenizer
import numpy as np
import argparse

tokenizers = {"bert":BertTokenizer,"roberta":RobertaTokenizer}

def get_data(args, data_type):
    path = args.data_path
    if data_type == "train":
        path = path + args.train_file
    elif data_type == "valid":
        path = path + args.valid_file
    else:
        path = path + args.test_file

    return pd.read_csv(path, sep='\t',header=None)

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

        self.data = get_data(args, data_type)
        self.device = args.device
        self.tokenizer = tokenizers[args.model_name].from_pretrained(args.tokenizer_name)


        self.tok_texts = self.data[0].apply((lambda x:
                                         self.tokenizer.encode(x,add_special_tokens=True))).values

        self.labels = self.data[1].values

        self.padded, self.mask= self._padding()

        self.dealDataset = DealDataset(self.padded,self.labels, self.mask, args)

    def _padding(self):

        max_len = 0
        for i in self.tok_texts:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [self.tokenizer.pad_token_id] * (max_len - len(i)) for i in self.tok_texts])

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


if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--model_name',type=str,default='bert',help='验证集文件名')
    parser.add_argument('--pretrained_model_name',type=str,default='bert-base-uncased',help='预训练模型')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='分词器')

    args = parser.parse_args()

    t = text_encoder_decoder(args)
    x = t.encode(["a b c d e f","this is a test"])
    y = t.decode(x)
    print(x)
    print(y)
