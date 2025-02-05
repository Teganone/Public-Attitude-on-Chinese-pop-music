import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels = {'past':0,'neutral':1,'now':2,'notmatch':3}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['analysis_result'] if label in labels]
        self.texts = [tokenizer(str(text), 
                                padding='max_length', 
                                max_length = 200, 
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
        # return self.texts,self.labels

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):
  # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
      # 进度条函数tqdm
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                
        # 通过模型得到输出
                output = model(input_id, mask)
                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                # print('--------train-------')
                # print('output:',output.argmax(dim=1))
                # print('train_label:',train_label)
                # print('acc:',acc)
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
                for val_input, val_label in val_dataloader:
          # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)
                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    # print('-----------dev------------')
                    # print('outputarg:',output.argmax(dim=1))
                    # print('output:',output)
                    # print('val_label:',val_label)
                    # print('acc:',acc)
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')           


def evaluate(model, test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    # print(test_dataloader)
    # print('item:',test_dataloader.getitem())
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
            #   print('test_label:',test_label)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label).sum().item()
            #   print('output:',output.argmax(dim=1))
            #   print('test_label:',test_label)
            #   print('acc:',acc)
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    

if __name__ == '__main__':
    df = pd.read_csv('./data/sentiment/sentiments_comment_label_1.csv')
    np.random.seed(112)

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])
    # print(df['content'])
    print('len:',len(df_train),len(df_val), len(df_test))
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-7
    train(model, df_train, df_val, LR, EPOCHS)
    evaluate(model, df_test)