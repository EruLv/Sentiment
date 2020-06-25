import torch
from torch import nn
import argparse
from preparing_data import dataset
import numpy as np

def predict(outputs):

    preds = torch.argmax(outputs,dim=1) #每一行的最大值下标

    return preds


def test(model, data_loader,args):
    hit = 0
    total = 0
    res_preds = []
    res_labels = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):

            inputs, labels, masks = data
            outputs = model(inputs, attention_mask=masks)
            preds = predict(outputs)

            res_labels.extend(list(labels.cpu().numpy()))
            res_preds.extend(list(preds.cpu().numpy()))

            hit += sum(labels == preds).item()
            total += len(labels)

        acc = hit/total     #准确率

        label_class = list(set(res_labels))
        P = []
        R = []
        res_labels = np.array(res_labels)
        res_preds = np.array(res_preds)
        for c in label_class:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(res_labels):
                p = res_preds[i]
                if(p==c and r == c):
                    tp += 1
                elif(p == c and r != c ):
                    fp += 1
                elif( p != c and r != c):
                    tn += 1
                elif( p != c and r == c):
                    fn += 1
            P.append(tp/(tp + fp))
            R.append(tp/(tp + fn))
        Macro_P = np.mean(P)
        Macro_R = np.mean(R)
        Macro_F = 2 * Macro_P*Macro_R / (Macro_P + Macro_R)

        print("准确率: {}, 平均精确率: {}, 平均召回率: {}, 平均F值: {}"
              .format(acc, Macro_R, Macro_R, Macro_F))



def main(args):
    model  = torch.load(args.model_path).to(args.device)
    test_data = dataset('test',args)
    test_data_loader = test_data.get_data_loader()
    test(model,test_data_loader, args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=8,help = '每批数据的数量')
    parser.add_argument('--model_name',type=str,default='roberta',help = '模型类别')
    parser.add_argument('--model_path',type=str,default='roberta_model_1.pt',help = '模型位置')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='分词器')
    parser.add_argument('--data_path', type=str, default='data/SST2/SST2/', help='数据集路径')
    parser.add_argument('--test_file', type=str, default='test.tsv', help='测试集文件名')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)
    args = parser.parse_args()

    main(args)
