import torch
from torch import nn
import argparse
from preparing_data import dataset
import numpy as np


def predict_single(outputs):
    preds = torch.argmax(outputs,dim=1) #每一行的最大值下标
    return preds

class bagging_models():
    def __init__(self, model_list, args):
        self.device = args.device
        self.model_list = model_list
        self.args = args
        self.models = []
        for name in model_list:
            self.models.append(torch.load(name).to(args.device))

    def __call__(self, inputs, attention_mask):
        outputs = [model(inputs, attention_mask=attention_mask)
                   for model in self.models]
        preds = np.array([predict_single(o).cpu().numpy() for o in outputs])
        bagging_result = []
        for i in range(preds.shape[1]):
            bagging_result.append(np.argmax(np.bincount(preds[:,i])))

        return torch.from_numpy(np.array(bagging_result)).to(self.device)


def test(model, data_loader,args):
    hit = 0
    total = 0
    for i, data in enumerate(data_loader):
        inputs, labels, masks = data
        outputs = model(inputs, attention_mask=masks)

        hit += sum(labels == outputs).item()
        print(hit)
        total += len(labels)
    acc = hit/total
    print("平均准确率为: {}".format(acc))


def main(args):
    model_list = ["roberta_model_1.pt","roberta_model_2.pt"]

    model  = bagging_models(model_list, args)

    test_data = dataset('test',args)
    test_data_loader = test_data.get_data_loader()

    test(model,test_data_loader, args)




if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=8,help = '每批数据的数量')
    parser.add_argument('--model_path',type=str,default='model.pt',help = '模型位置')
    parser.add_argument('--model_name',type=str,default='roberta',help='预训练模型名')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='分词器')
    parser.add_argument('--data_path', type=str, default='data/SST2/SST2/', help='数据集路径')
    parser.add_argument('--test_file', type=str, default='test.tsv', help='测试集文件名')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)
    args = parser.parse_args()

    main(args)
