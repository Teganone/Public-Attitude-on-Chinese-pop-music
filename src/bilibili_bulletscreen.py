import re
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import csv

allvideos_info = './data/video_info.csv'
# video_print_info = './data/video_print_info.txt'
onevideo_url='https://bilibili.com/video/{}'.format('BV1dJ411M7RV')

def get_all_videos(onevideo_url):
    headers = {'user-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',}
    print("the address of video: ",onevideo_url)
    r1 = requests.get(onevideo_url,headers)
    r1.encoding = r1.apparent_encoding
    html1 = r1.text
    # print(html1)
    # file = open(video_print_info,'w')
    # file.write(html1)
    cid_list = re.findall('"desc_v2":null,"is_chargeable_season":false,"is_blooper":false},"page":{"cid":(.*?),"page":1,"from":"vupload","part":"',html1)
    bvid_list = re.findall('"dimension":{"width":1920,"height":1080,"rotate":0}},"bvid":"(.*?)"}',html1)
    title_list = re.findall(',"page":1,"from":"vupload","part":"(.*?)","duration":',html1)
    df1 = pd.DataFrame()
    df1['bvid'] = bvid_list[1:32]
    df1['cid'] = cid_list[1:32]
    df1['title'] =title_list[1:32]
    dfheader1 = ['bvid','cid','title']
    # csv_file = './data/bvidcidtest2.0.csv'
    df1.to_csv(allvideos_info,encoding='utf_8_sig',mode='w',index=False,header=dfheader1)


# def get_bilibili_bullet(video_url,recv_file):
    
#     bulletsrceen_url = 'https://comment.bilibili.com/{}.xml'.format(cid[18])
#     print(bulletsrceen_url)
#     # bulletsrceen_url = 'https://comment.bilibili.com/94942381.xml'
#     res = requests.get(bulletsrceen_url)
#     xml = res.text.encode('raw_unicode_escape')
#     soup = BeautifulSoup(xml,"xml")
#     bullet_list =  soup.find_all("d")
#     # print(bullet_list)
#     print("fetch bullet screen {} ".format(len(bullet_list)))
#     bullet_url_list =[]
#     time_list = []
#     text_list = []
#     for bullet in bullet_list:
#         data_split = bullet['p'].split(',')
#         temp_time = time.localtime(int(data_split[4]))
#         time_list.append(time.strftime("%Y-%m-%d %H:%M:%S",temp_time))
#         text_list.append(bullet.text)
#         bullet_url_list.append(bulletsrceen_url)

#     df = pd.DataFrame()
#     df['bullet_url'] = bullet_url_list
#     df['bullet_time'] = time_list
#     df['bullet_content'] = text_list
#     dfheader = ['bullet_url','bullet_time','bullet_content']

#     year = 2009
#     csv_file = './data/chinese_music_year_{}.csv'.format(year)

#     df.to_csv(csv_file,encoding='utf_8_sig',mode='w',index=False,header=dfheader)
#     print('DONE!')



def get_bilibili_bullet(cid,title,recv_file):
    
    bulletsrceen_url = 'https://comment.bilibili.com/{}.xml'.format(cid)
    res = requests.get(bulletsrceen_url)
    xml = res.text.encode('raw_unicode_escape')
    soup = BeautifulSoup(xml,"xml")
    bullet_list =  soup.find_all("d")
    print("fetch bullet screen {} ".format(len(bullet_list)))
    bullet_url_list =[]
    time_list = []
    text_list = []
    title_list = []
    for bullet in bullet_list:
        data_split = bullet['p'].split(',')
        temp_time = time.localtime(int(data_split[4]))
        time_list.append(time.strftime("%Y-%m-%d %H:%M:%S",temp_time))
        text_list.append(bullet.text)
        bullet_url_list.append(bulletsrceen_url)
        title_list.append(title)

    df = pd.DataFrame()
    # df['bullet_url'] = bullet_url_list
    df['video_tile'] = title_list
    df['bullet_time'] = time_list
    df['bullet_content'] = text_list
    # dfheader = ['bullet_url','bullet_time','bullet_content']
    dfheader = ['video_title','bullet_time','bullet_content']

    # csv_file = './data/chinese_music_year_{}.csv'.format(year)

    df.to_csv(recv_file,encoding='utf_8_sig',mode='w',index=False,header=dfheader)
    # print('Year ',year,' DONE!')




if __name__ == '__main__':
    print('search start...')
    # year = 2009
    # csv_file = './data/chinese_music_year_{}.csv'.format(year)
    
    # get_bilibili_bullet(video_url='https://bilibili.com/video/{}'.format('BV1dJ411M7RV'),recv_file=file)
    get_all_videos(onevideo_url)
    # with open(allvideos_info) as csvfile:
    #     reader = csv.reader(csvfile)
    #     column_cid = [row[1] for row in reader][1:]
    #     column_title = [row[0] for row in reader]
    #     print(column_title)
    csv_file = pd.read_csv(allvideos_info)
    column_cid = csv_file['cid']
    column_title = csv_file['title']
    print(column_title)
    for cid, title, year in zip(column_cid,column_title,range(2020,2021)):
        recv_file = './data/bilibilibullet/chinese_music_year_{}.csv'.format(year)
        get_bilibili_bullet(cid,title,recv_file)
        print('year ',year," done!")
    



    