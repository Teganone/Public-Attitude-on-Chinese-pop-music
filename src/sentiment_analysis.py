import pandas as pd
from snownlp import SnowNLP
from pprint import pprint
import jieba.analyse
from PIL import Image
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections


def sentiment_analyse(bullet_list):
    score_list = []
    tag_list = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    for comment in bullet_list:
        tag = ''
        sentiments_score = SnowNLP(comment).sentiments
        if sentiments_score < 0.3:
            tag = 'negative'
            neg_count += 1
        elif sentiments_score > 0.7:
            tag = 'positive'
            pos_count += 1
        else: 
            tag = 'netural'
            neu_count += 1
        score_list.append(sentiments_score)
        tag_list.append(tag)
    sentiment_sum.write('Ratio of positive comments: {}\n'.format(round(pos_count / (pos_count + neg_count + neu_count), 4)))
    # print('Ratio of positive comments: ',round(pos_count / (pos_count + neg_count + neu_count), 4))
    sentiment_sum.writelines('Ratio of negative comments: {}\n'.format(round(neg_count / (pos_count + neg_count + neu_count), 4)))
    sentiment_sum.writelines('Ratio of neutral comments: {}\n'.format(round(neu_count / (pos_count + neg_count + neu_count), 4)))

    df['sentiment_score'] = score_list
    df['analysis_result'] = tag_list
    # df.to_excel('./data/sentiments_outputs.xlsx',index=None)
    df.to_csv('./data/sentiment/sentiments_comment_{}.csv'.format(year),index=None)
    print('done')



def make_wordcloud(input_str,stopwords,outfile):
    print('start generating wordcloud: {}'.format(outfile))
    try:
        # stopwords = stopwords
        # background_Image = np.array(Image.open('wo'))
        wordcloud = WordCloud(
            background_color='white',
            width=1500,
            height=1200,
            max_words=1000,
            font_path='./27891586950.ttf',
            stopwords=stopwords,
            max_font_size=99,
            min_font_size=16,
            random_state=50
        )
        # input_str = dict(collections.Counter(input_str))
        # jieba_text = ' '.join(jieba.lcut(input_str))
        # wordcloud.generate_from_frequencies(input_str)
        wordcloud.generate(input_str)
        # print('1111')
        # plt.imshow(wordcloud,interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
        # print(111)
        wordcloud.to_file(outfile)
        print('save wordcloud picture:{}'.format(outfile))

    except Exception as e:
        print('make_wordcloud except:{}'.format(e))


if __name__ == '__main__':
    # df = pd.read_csv('./data/chinese_music_year_2020.csv')
    # bullet_list = df['bullet_content'].values.tolist()
    # print('length of the bullet_list is:{}'.format(len(bullet_list)))
    # bullet_list = [str(bullet) for bullet in bullet_list]
    # bullet_str = ' '.join(str(bullet) for bullet in bullet_list)

    # sentiment_analyse(bullet_list)

    # keywords_top10 = jieba.analyse.extract_tags(bullet_str,withWeight=True,topK=10)
    # print('top10 key words:',keywords_top10)
    # pprint(keywords_top10) 
    # # stopwords = ['的','啊','她','是','了','你','我','都','也','不','在','吧','说','就是','这','有','，','。','！','？',' ']
    # outfile = './data/wordcloud.jpg'
    # with open('./data/stopwords/hit_stopwords.txt','r',encoding='utf-8') as f:
    #         stopwords = f.readlines()
    #         stopwords = [l.strip() for l in stopwords]
    # handcraftedstopwords = ['哈哈','哈哈哈','这首','有','没有','真的','想起','老婆','老公']
    # stopwords = stopwords + handcraftedstopwords
    # make_wordcloud(bullet_str,stopwords,outfile)

    with open('./data/stopwords/all_stopwords.txt','r',encoding='utf-8') as f:
                stopwords = f.readlines()
                stopwords = [l.strip() for l in stopwords]
    # print("first:",stopwords)
    handcraftedstopwords = ['啊啊啊',' ',' ','草','握','哈哈','哈哈哈','这首','有','没有','真的','想起','老婆','老公','这歌','卧槽','首歌','一年','听']
    stopwords = stopwords + handcraftedstopwords
    # print("second:",stopwords)
    
    #=========bullet=========

    # for year in range(1991,2022):
    #     df = pd.read_csv('./data/bilibilibullet/chinese_music_year_{}.csv'.format(year))
    #     wordcloud_outfile = './data/wordcloud/wordcloud_music_year_{}.jpg'.format(year)
    #     sentiment_sum = open('./data/sentiment/bullet_sentiment_sum.log','a+')
    #     keywords_top10_file = open('./data/bullet_keywords_top10_file.txt','a')
    #     sentiment_sum.writelines('\n============sentiment sum of {}=============\n'.format(year))
    #     bullet_list = df['bullet_content'].values.tolist()
    #     print(year,':length of the bullet_list is:{}'.format(len(bullet_list)))
    #     bullet_list = [str(bullet) for bullet in bullet_list]
    #     bullet_str = ' '.join(str(bullet) for bullet in bullet_list)
    #     bullet_words = ','.join(jieba.lcut(bullet_str))
    #     bullet_word_list = list(set(bullet_words.split(',')))
    #     # print(bullet_word_list)
    #     for bullet_word in bullet_word_list:
    #         # print(bullet_word)
    #         if bullet_word in stopwords:
    #             bullet_word_list.remove(bullet_word)
    #     sentiment_analyse(bullet_list)
    #     bullet_str_new = ' '.join(str(bullet) for bullet in bullet_word_list)

    #     bullet_str = jieba.analyse.set_stop_words(stopwords)
    #     keywords_top10 = jieba.analyse.extract_tags(bullet_str_new,withWeight=True,topK=10)

    #     print('top10 key words:')
    #     pprint(keywords_top10) 
    #     keywords_top10_file.write('-----------{}------------\n'.format(year))
    #     for word in keywords_top10:
    #         keywords_top10_file.write('{}\n'.format(word))
    #     make_wordcloud(bullet_str,stopwords,wordcloud_outfile)


    #========comment============


    year_list = [2005,2006,2012,2013,2020,2021]
    for year in year_list:
        df = pd.read_csv('./data/bilibilicomment/bilibilicomment_{}.csv'.format(year))
        wordcloud_outfile = './data/wordcloud/wordcloud_music_comment_{}.jpg'.format(year)
        sentiment_sum = open('./data/sentiment/comment_sentiment_sum.log','a+')
        keywords_top10_file = open('./data/comment_keywords_top10_file.txt','a')
        sentiment_sum.write('============sentiment sum of {}=============\n'.format(year))
        bullet_list = df['content'].values.tolist()   
        print(year,':length of the comment_list is:{}'.format(len(bullet_list)))
        bullet_list = [str(bullet) for bullet in bullet_list]
        bullet_str = ' '.join(str(bullet) for bullet in bullet_list)
        bullet_words = ','.join(jieba.lcut(bullet_str))
        bullet_word_list = list(set(bullet_words.split(',')))
        # print(bullet_word_list)
        for bullet_word in bullet_word_list:
            # print(bullet_word)
            if bullet_word in stopwords:
                bullet_word_list.remove(bullet_word)
        sentiment_analyse(bullet_list)
        bullet_str_new = ' '.join(str(bullet) for bullet in bullet_word_list)

        # bullet_str = jieba.analyse.set_stop_words(stopwords)
        keywords_top10 = jieba.analyse.extract_tags(bullet_str_new,withWeight=True,topK=10)

        print('top10 key words:')
        pprint(keywords_top10) 
        keywords_top10_file.write('-----------{}------------\n'.format(year))
        for word in keywords_top10:
            keywords_top10_file.write('{}\n'.format(word))
        make_wordcloud(bullet_str,stopwords,wordcloud_outfile)