import re
import pandas as pd
import os
import codecs
# str = 'age":1,"from":"vupload","part":"【神仙打架】回顾1999年华语乐坛巅峰之曲，至今依然是KTV金曲！","duration":989,"vid":"","weblink":"","dimension":{"width":1920,"height":1080,"rotate":0}},"bvid":"BV1zJ411g7xQ"},{"season_id":48627,"section_id":57309,"id":715566,"aid":68868300,"cid":119348970,"title":"2000年华语乐坛经典回顾","attribute":0,"arc":{"aid":68868300,"videos":0,"type_id":0,"type_name":"","copyright":0,"pic":"http:\u002F\u002Fi1.hdslb.com\u002Fbfs\u002Farchive\u002Fb90312cadef0cf34cd86b0e9287e908ce7f77004.jpg","title":"2000年华语乐坛经典回顾","pubdate":1569329304,"ctime":1569329304,"desc":"","state":0,"duration":985,"rights":{"bp":0,"elec":0,"download":0,"movie":0,"pay":0,"hd5":0,"no_reprint":0,"autoplay":0,"ugc_pay":0,"is_cooperation":0,"ugc_pay_preview":0,"arc_pay":0,"free_watch":0},"author":{"mid":0,"name":"","face":""},"stat":{"aid":68868300,"view":303186,"danmaku":7756,"reply":302,"fav":3824,"coin":3242,"share":1232,"now_rank":0,"his_rank":0,"like":6557,"dislike":0,"evaluation":"","argue_msg":""},"dynamic":"","dimension":{"width":0,"height":0,"rotate":0},"desc_v2":null,"is_chargeable_season":false,"is_blooper":false},"page":{"cid":119348970,"page":1,"from":"vupload","part":"【神仙打架】回顾2000年华语乐坛“疯狂”之路，全程都能跟着哼唱！","duration":985,"vid":"","weblink":"","dimension":{"width":1920,"height":1080,"rotate":0}},"bvid":"BV1dJ411M7RV"},{"season_id":48627,"section_id":57309,"id":715567,"aid":69068008,"cid":119696605,"title":"2001年华语乐坛经典回顾","attribute":0,"arc":{"aid":69068008,"videos":0,"type_id":0,'
# # str = '"dimension":{"width":1920,"height":1080,"rotate":0}},"bvid":"BV1zJ411g7xQ"},{"season_id":48627,"section_id":57309,"id":715566,"aid":68868300,"cid":119348970,"title":'
# test = re.findall('"dimension":{"width":1920,"height":1080,"rotate":0}},"bvid":"(.*?)"}',str)
# tets2 = re.findall(',"cid":(.*?),"title":',str)
# print(test)
# print(tets2)


# for i, year in zip(range(0,22), range(2000,2022)):
#     # path = './data/chinese_music_year_{}.csv'.format(year) 
#     print(i,year)
#     # file = open(path,'w')
#     # file.write('test')


# with open('./data/stopwords/hit_stopwords.txt','r',encoding='utf-8') as f:
#             stopwords = f.readlines()
#             stopwords = [l.strip() for l in stopwords]
# print(stopwords)



# year = 1991
# df = pd.read_csv('./data/chinese_music_year_{}.csv'.format(year))
# # wordcloud_outfile = './data/wordcloud/wordcloud_music_year_{}.jpg'.format(year)
# bullet_list = df['bullet_content'].values.tolist()
# print(type(bullet_list[0]))
# print(bullet_list[0])
# print(year,':length of the bullet_list is:{}'.format(len(bullet_list)))
# bullet_list = [str(bullet) for bullet in bullet_list]
# print(type(bullet_list[0]))
# print(bullet_list[0])
# bullet_str = ' '.join(str(bullet) for bullet in bullet_list)
# print(type(bullet_str))
# print(bullet_str)


# with open('./data/stopwords/hit_stopwords.txt','r',encoding='utf-8') as f:
#             stopwords = f.readlines()
#             stopwords = [l.strip() for l in stopwords]
    

# wdir = './data/stopwords/'
# desfile = open('./data/stopwords/all_stopwords.txt','a+')
# for num, file in enumerate(sorted(os.listdir(wdir))):
#     try:
#         print(num,file)
#         openfile = codecs.open(wdir + file, 'r', encoding='utf-8', errors='ignore')
#         # print(openfile)
#         stopwords = openfile.readlines()
#         stopwords = [l.strip() for l in stopwords]
#         print(stopwords)
#         for stopword in stopwords:
#             stopword = stopword+'\n'
#             desfile.write(stopword)
#     except:
#             print('file {} skipped'.format(num))
# print(list(set(stopwords)))


#服务器看不了 没有ui
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = './danmaku.ttf'
# plt.rcParams['axes.unicode_minus']=False
# x = [0.4433, 0.1111, 0.4456]
# plt.subplot(131)
# plt.title('正常')
# plt.pie(x)
# plt.subplot(132)
# plt.title('添加labels')
# plt.pie(x,labels=['positive','negative','neutral'],autopct='%1.1f%%')
# # labeldistance默认为是1.1
# plt.subplot(133)
# plt.title('添加labels和labeldistance')
# plt.pie(x,labels=['positive','negative','neutral'],labeldistance=1.2)
# plt.show()

# list = list(range(0,10))
# print(list)
# print(id(list))
# list.clear()
# print(list)
# print(id(list))
# list = []
# print(id(list))


# year_list = [2005,2006,2012,2013,2020,2021]
# for year in year_list:
#     filepath = f'./data/bilibilicomment/bilibilicomment_{year}.csv'
#     print(filepath)
#     file = open(filepath,'a')
    
# import time
# import random
# for i in range(0,5) :
#     time1=random.uniform(1,5)
#     print(time1)
#     time.sleep(time1)


# import pandas as pd
 
# # 读取文件
# data = pd.read_csv('./data/bilibilicomment/bilibilicomment_label.csv')
 
# # 原来的值：新的值
# # 使用map属性 
# change= {'positive':'怀旧', 'negative':'判今','nautral':'中立'}
# data['sentiment'] = data['sentiment'].map(change)
 
# #生成新的文件
# data.to_csv("./data/bilibilicomment/bilibilicomment_label.csv")


# import re

# file = open()
# number = re.findall("\d+",)    # 输出结果为列表
# print(number)
 
# # 输出结果：['12', '333', '4']
