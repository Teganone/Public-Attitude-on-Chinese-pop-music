import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/sentiment/bullet_sentiment_sum.csv')
# data = pd.read_csv('./data/sentiment/comment_sentiment_sum.csv')
# data = pd.read_csv('./data/BertClassifier/sentiment/comment_sentiment_sum.csv')
year_list = data['year'].values.tolist()
pos_list = data['positive'].values.tolist()
neu_list = data['neutral'].values.tolist()
neg_list = data['negative'].values.tolist()

fig = plt.figure(figsize=(7,4),dpi=200)
ax1 = fig.add_subplot(111)
line1, = ax1.plot(year_list,pos_list,'r:',lw=1,label='positive')
line2, = ax1.plot(year_list,neu_list,'k-',lw=1,label='neutral')
line3, = ax1.plot(year_list,neg_list,'-',color='steelblue',label='negative')
plt.legend((line1,line2,line3),('positive','neutral','negative'),loc='center left',frameon=False,framealpha=0.5)
ax1.set_xlabel('year')
ax1.set_ylabel('percent:(%)')
plt.title('timeline of sentiment')
plt.show()