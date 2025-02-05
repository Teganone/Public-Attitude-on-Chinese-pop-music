# here put the import lib
import argparse
from sklearn.metrics import mean_squared_error, r2_score
import torch
import json
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import collections
import pandas as pd 
import os
import sys
import random
import pickle
import warnings 
warnings.filterwarnings("ignore")
import datetime

# labels = {'怀旧':0,'中立':1,'判今':2}
labels = {'positive':0,'netural':1,'negative':2}


# train_set = NewsDataset(train,tokenizer,wordKept=wordKept,stopwords=stopwords)
# class Dataset(Dataset):
#     """
#     Dataset class for News-Stock Prediction Task
#     """
#     def __init__(self,
#                  df,
#                  tokenizer,
#                  stopwords=None,
#                 ):

#         # print(df['sentiment'])
        
#         self.labels = [labels[sentiment] for sentiment in df['analysis_result'] if sentiment in labels]
    
#         self.texts = [tokenizer(text,
#                                 padding = 'max_length',
#                                 max_length = 300,
#                                 truncation = True,
#                                 return_tensors = 'pt')
#                                 for text in df['content']]
#         # print(self.texts)
#         # # print(self.texts.index)
#         # print(self.texts)

#     def classes(self):
#         return self.labels
    
#     def __len__(self):
#         return len(self.labels)

#     def get_batch_labels(self,index):
#         return np.array(self.labels[index])

#     def get_batch_texts(self,index):
#         return self.texts[index]

#     def __getitem__(self, index):
#         batch_texts = self.get_batch_texts(index)
#         batch_y = self.get_batch_labels(index)
#         return batch_texts,batch_y
        
    
#     def _filter_by_stopwords(self,wordLists,stopwords):
#         filterd_lists = []
#         for wordList in wordLists:
#             w_array = pd.Series(wordList)
#             filterd_lists.append(list(w_array[~w_array.isin(stopwords)]))
#         return filterd_lists

class Dataset(Dataset):
    def __init__(self, df,tokenizer):
        self.labels = [labels[label] for label in df['analysis_result']]
        self.texts = [tokenizer(text, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors='pt') 
                      for text in df['content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# class BERTClassifier(nn.Module):
#     def __init__(self, embed_size=768):
#         super(BERTClassifier,self).__init__()
#         self.embedding = BertModel.from_pretrained('bert-base-chinese')
#         self.decoder = nn.Linear(embed_size,3)
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()

#     def forward(self,inputs,att_mask):
#         embeddings,pooled_out = self.embedding(inputs,att_mask,return_dict=False)
#         dropout_output = self.dropout(pooled_out)
#         linear_output = self.decoder(dropout_output)
#         final_layer = self.relu(linear_output)
#         return final_layer
#         # outs = self.decoder(self.dropout(embeddings)).flatten()
#         # return outs

class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer



#应用:training(train_iter, dev_iter,net, loss, optimizer, device, num_epochs,early_stopping,eval_interval,writer)

def training(train_iter, dev_iter, model, loss, optimizer, device, epochs):
    #to(device):将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    model = model.to(device)
    batch_count = 0
    break2 = False
        # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
      # 进度条函数tqdm
        for train_input, train_label in tqdm(train_iter):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
    # 通过模型得到输出
            output = model(input_id, mask)
            print("train:",output)
            # 计算损失
            batch_loss = loss(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
    # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
    # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in dev_iter:
        # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                print('dev:',output)
                batch_loss = loss(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(df_train): .3f} 
            | Train Accuracy: {total_acc_train / len(df_train): .3f} 
            | Val Loss: {total_loss_val / len(df_dev): .3f} 
            | Val Accuracy: {total_acc_val / len(df_dev): .3f}''')           
    
def evaluate(model, test_iter):

    
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_iter:
              test_label = test_label.to(device)
              print('test_label:',test_label)
              mask = test_input['attention_mask'].to(device)
              print('mask:',mask)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              print('input_id:',input_id)
              output = model(input_id, mask)
              print('output:',output)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              print(output.argmax(dim=1))
              print('acc:',acc)
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')


if __name__ == '__main__':


    # USE_MULTI_GPU = True
    # if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    #     MULTI_GPU = True
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  #设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
    #     device_ids = [0,1]
    # else:
    #     MULTI_GPU = False


    random.seed(123)
    #torch.device代表将torch.Tensor分配到的设备的对象。torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # print("[{}] Model Config:".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


    # # ### LOAD DATA 
    # print("[{}] LOADING DATA...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    


    print("--------------------- DONE!")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # df = pd.read_csv('./data/bilibilicomment/bilibilicomment_label.csv')
    df = pd.read_csv('./data/sentiment/sentiments_comment_label.csv')
    # df = pd.read_csv('./data/sentiment/sentiments_comment_label.csv')

    # train_content_list = train_file['content'].values.tolist()
    # # print(train_list)
    # train_sentiment_list = train_file['sentiment'].values.tolist()
    # train_set = dict(zip(train_content_list,train_sentiment_list))
    # print(train_set)


    # print(df)
    # print(df['content'])
    # print(df['sentiment'])
    df_train, df_dev, df_test = np.split(df.sample(frac=1,random_state=42),[int(.8*len(df)),int(.9*len(df))])
    # print(len(df_train))
    # print(len(df_dev))
    # print(len(df))
    # print(df_dev['content'])
    # print(df_dev['sentiment'])
    

    
    print("[{}] CONSTRUCTING TRAINING SET...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    train_set = Dataset(df_train,tokenizer)
    print("--------------------- DONE!")

    print("[{}] CONSTRUCTING VALIDATION SET...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    dev_set = Dataset(df_dev,tokenizer)
    print("--------------------- DONE!")

    print("[{}] CONSTRUCTING TEST SET...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    test_set = Dataset(df_test,tokenizer)
    print("--------------------- DONE!")
    
    eval_interval = 10
    BATCH_SIZE = 2


    train_iter = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    dev_iter = DataLoader(dev_set,batch_size=BATCH_SIZE,shuffle=True)
    test_iter = DataLoader(test_set,batch_size=BATCH_SIZE)
    
   
    

    embed_size = 768
    net = BERTClassifier(embed_size)
    lr = 0.00005
    num_epochs = 2

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    print(loss)
    if use_cuda:
        net = net.cuda()
        loss = loss.cuda()



    print("[{}] TRAINING...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    training(train_iter, dev_iter,net, loss, optimizer, device, num_epochs)

    print("[{}] EVALUATING...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    evaluate(net,test_iter)

   