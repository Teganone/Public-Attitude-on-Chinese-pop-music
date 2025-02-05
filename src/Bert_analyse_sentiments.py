import BertClassifier_sentiments
import pandas as pd
import torch

newlabels = {0:'positive',1:'neutral',2:'negative'}
def classifing(df_data,outputfile):
    bertlabel = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    test = BertClassifier_sentiments.Dataset(df_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    model = BertClassifier_sentiments.BertClassifier()
    # print(test_dataloader)
    # print('item:',test_dataloader.getitem())
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
            #   print('test_label:',test_label)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
            #   print(output)
              labels = output.argmax(dim=1)
            #   print('before:',labels)
              labels = labels.cpu().numpy().tolist()
            #   print('after:',labels)
            #   print(type(labels))
              for label in labels:
                    label = newlabels[label]
                    bertlabel.append(label)
                    # print("label:",label)
                    if label == 'positive':
                        pos_count += 1
                    if label == 'neutral':
                        neu_count += 1
                    if label == 'negative':
                        neg_count += 1
    
    df_data['bertlabel'] = bertlabel
    df_data.to_csv(outputfile,index=None)
    sentiment_year_list.append(year)
    sentiment_pos_list.append((round(pos_count / (pos_count + neg_count + neu_count), 4)))
    sentiment_neg_list.append((round(neg_count / (pos_count + neg_count + neu_count), 4)))
    sentiment_neu_list.append((round(neu_count / (pos_count + neg_count + neu_count), 4)))
    print('done')


            

if __name__ == '__main__':
    sentiment_year_list = []
    sentiment_pos_list = []
    sentiment_neg_list = []
    sentiment_neu_list = []
    for year in range(2000,2022):
        df_data = pd.read_csv('./data/sentiment/sentiments_comment_{}.csv'.format(year))
        outputfile = './data/BertClassifier/sentiment/bertclassifier_comment_{}.csv'.format(year)
        sentiment_sum_file = open('./data/BertClassifier/sentiment/comment_sentiment_sum.csv','a+')
        print('-----{}-------'.format(year))
        classifing(df_data,outputfile)

    sentiment_sum_df = pd.DataFrame()
    sentiment_sum_df['year'] = sentiment_year_list
    sentiment_sum_df['positive'] = sentiment_pos_list
    sentiment_sum_df['neutral'] = sentiment_neu_list
    sentiment_sum_df['negative'] = sentiment_neg_list
    sentiment_sum_df.to_csv(sentiment_sum_file,encoding='utf_8_sig',mode='w',index=False)
    # sentiment_sum_df.to_csv('./data/sentiment/bullet_sentiment_sum.csv',encoding='utf_8_sig',mode='w',index=False)
    print('sentiment done!')
    print('bert classify done!')