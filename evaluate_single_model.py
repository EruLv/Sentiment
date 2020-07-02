import torch
import argparse
from preparing_data import DataSet
import numpy as np
from tqdm import tqdm

def predict(outputs):
    '''
    Descriptions: 预测是第几类
    Args:
        outputs: 模型的logits,输出
    Returns:
        preds:最大概率对应下标
    '''
    preds = torch.argmax(outputs,dim=1) #每一行的最大值下标
    return preds


def test(model, data_loader, args):
    '''
    Descriptions:
        测试模型的4个指标acc,precision,recall,F1
        不把聚类生成的类合并
    Args:
        model:已经训练好的模型
        data_loader: 测试数据的迭代器
    '''
    hit = 0
    total = 0
    res_preds = []
    res_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

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
                if(p == c and r == c):
                    tp += 1
                elif(p == c and r != c ):
                    fp += 1
                elif( p != c and r != c):
                    tn += 1
                elif( p != c and r == c):
                    fn += 1
            if tp != 0:
                P.append(tp/(tp + fp))
                R.append(tp/(tp + fn))
            else:
                P.append(0)
                R.append(0)
        Macro_P = np.mean(P)
        Macro_R = np.mean(R)
        Macro_F = 2 * Macro_P*Macro_R / (Macro_P + Macro_R)

        print("准确率: {}, 平均精确率: {}, 平均召回率: {}, 平均F值: {}"
              .format(acc, Macro_R, Macro_R, Macro_F))

def test_ingored_cluster(model, data_loader, args):
    '''
    Descriptions:
        测试模型的4个指标acc,precision,recall,F1
        把聚类生成的类合并，比如，1聚类成为1,2,3,4类，测试时讲1,2,3,4和并为1类
    Args:
        model:已经训练好的模型
        data_loader: 测试数据的迭代器
    '''
    res_preds = []
    res_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

            inputs, labels, masks = data
            outputs = model(inputs, attention_mask=masks)
            preds = predict(outputs)
            res_labels.extend(list(labels.cpu().numpy()))
            res_preds.extend(list(preds.cpu().numpy()))

        label_class = [0,1,5]
        P = []
        R = []
        res_labels = np.array(res_labels)
        res_preds = np.array(res_preds)
        acc = 0
        for c in label_class:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(res_labels):
                p = res_preds[i]
                if p == 1 or p == 2 or p == 3 or p == 4:
                    p = 1
                if r == 1 or r == 2 or r == 3 or r == 4:
                    r = 1
                if(p == c and r == c):
                    tp += 1
                elif(p == c and r != c ):
                    fp += 1
                elif( p != c and r != c):
                    tn += 1
                elif( p != c and r == c):
                    fn += 1

            acc += (tp + tn)/(tp + fn + fp + tn)

            if tp != 0:
                P.append(tp/(tp + fp))
                R.append(tp/(tp + fn))
            else:
                P.append(0)
                R.append(0)
        Macro_P = np.mean(P)
        Macro_R = np.mean(R)
        Macro_F = 2 * (Macro_P *Macro_R )/ (Macro_P + Macro_R)
        acc = acc / 3
        print("准确率: {}, 平均精确率: {}, 平均召回率: {}, 平均F值: {}"
              .format(acc, Macro_P, Macro_R, Macro_F))

def main(args):

    model  = torch.load(args.model_path).to(args.device)    #加载模型
    test_data = DataSet('test',args)    #加载测试集数据集
    test_data_loader = test_data.get_data_loader()  #获取测试集的迭代器
    test(model,test_data_loader, args)  #测试指标
    # test_ingored_cluster(model,test_data_loader, args)    #测试指标，合并聚类生成的类

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=8,help = '每批数据的数量')
    parser.add_argument('--model_name',type=str,default='bert',help = '模型类别，中文统统用bert')
    parser.add_argument('--model_path',type=str,default='models/last_model_1.pt',help = '模型存储位置')
    parser.add_argument('--tokenizer_name', type=str, default='models/chinese-roberta-wwm-ext', help='分词器')
    parser.add_argument('--data_path', type=str, default='data/raw/', help='数据集路径')
    parser.add_argument('--test_file', type=str, default='test.csv', help='测试集文件名')
    parser.add_argument('--max_len',type=int,default=128,help='最大补全长度')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type = str, default= device)

    args = parser.parse_args()

    main(args)
