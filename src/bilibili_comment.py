# 发送请求
import requests
# 将数据存入数据库
# import MySQLdb
# 每次请求停1s，太快会被B站拦截。
import time
import pandas as pd
import random
import json


# # 连接数据库
# conn = MySQLdb.connect(host="localhost", user='root', password='admin', db='scholldatabase', charset='utf8')
# cursor = conn.cursor()
# # 预编译语句
# sql = "insert into bilibili(rpid,root,name,avatar,content) values (%s,%s,%s,%s,%s)"

rpid_list = []
name_list = []
content_list = []
like_list = []
rcount_list = []

# 爬虫类（面向对象）
class JsonProcess:
    def __init__(self):
        self.Json_data = ''
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'accept': 'application/json, text/plain, */*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9'
        }
	
    # 发送爬取请求
    def spider(self, URL):
        url = URL
        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        # print(response.text)
        self.Json_data = response.json()['data']['replies']
        # self.Json_data = response.json()
        # print(self.Json_data)
        # print('1111')


# 爬取子评论
def getSecondReplies(root):
    reply = JsonProcess()
    # 页数
    pn = 1
    # 不知道具体有多少页的评论，所以使用死循环一直爬
    while pn<=20:
        url = f'https://api.bilibili.com/x/v2/reply/reply?jsonp=jsonp&pn={pn}&type=1&oid=979849123&ps=10&root={root}&_=1647581648753'
        # 每爬一次就睡1秒
        time.sleep(1)
        reply.spider(url)
        # 如果当前页为空（爬到头了），跳出子评论
        if reply.Json_data is None:
            break
        # 组装数据，存入数据库
        for node in reply.Json_data:
            rpid = node['rpid']
            name = node['member']['uname']
            # avatar = node['member']['avatar']
            content = node['content']['message']
            like = node['like']
            rcount = node['rcount']
            data = (rpid, root, name, content, like, rcount)
            rpid_list.append(rpid)
            name_list.append(name)
            content_list.append(content)
            like_list.append(like)
            rcount_list.append(rcount)
            # open()
            # try:
            #     # cursor.execute(sql, data)
            #     # conn.commit()
            # except:
            #     pass
            print(rpid, ' ', name, ' ', content, ' ', like ,' ', rcount, ' ', root)
        # 每爬完一次，页数加1
        pn += 1

        
# 爬取根评论
def getReplies(jp, i):
    # 不知道具体有多少页的评论，所以使用死循环一直爬
    while i<=200:
        url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=81278064&plat=1&seek_rpid=&type=1' #2019
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=76649759&plat=1&seek_rpid=&type=1' #2017
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=73509827&plat=1&seek_rpid=&type=1' #2009
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=71930768&plat=1&seek_rpid=&type=1'  #2007
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=70592690&plat=1&seek_rpid=&type=1' #2004
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=69539295&plat=1&seek_rpid=&type=1' #2002
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=74185864&plat=1&seek_rpid=&type=1' #2011
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=73773664&plat=1&seek_rpid=&type=1' #2010
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=70089003&plat=1&seek_rpid=&type=1' #2003
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=69068008&plat=1&seek_rpid=&type=1' #2001
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=68868300&plat=1&seek_rpid=&type=1' #2000
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=73177392&plat=1&seek_rpid=&type=1' #2008
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=75364815&plat=1&seek_rpid=&type=1' #2014
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=75798084&plat=1&seek_rpid=&type=1' #2016
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=75583531&plat=1&seek_rpid=&type=1' #2015
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=77250023&plat=1&seek_rpid=&type=1' #2018
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=550272309&plat=1&seek_rpid=&type=1' #2021
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=74573981&plat=1&seek_rpid=&type=1'  #2012
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=74887212&plat=1&seek_rpid=&type=1'  #2013
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=755512750&plat=1&seek_rpid=&type=1'  #2020
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next=1&oid=755512750&plat=1&seek_rpid=&type=1' #2020
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=56920510&plat=1&seek_rpid=&type=1'  #2005
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={i}&oid=71541567&plat=1&seek_rpid=&type=1' #2006
        # url = f'https://api.bilibili.com/x/v2/reply/main?jsonp=jsonp&next={i}&type=1&oid=979849123&mode=3&plat=1&_=1647577851745'
        jp.spider(url)
        # 如果当前页为空（爬到头了），跳出循环，程序结束。
        if jp.Json_data is None:
            break
        # print(jp.Json_data)

        # 组装数据，存入数据库。
        for node in jp.Json_data:
            print('===================')
            rpid = node['rpid']
            name = node['member']['uname']
            # avatar = node['member']['avatar'] 头像
            content = node['content']['message']
            like = node['like']
            rcount = node['rcount']
            data = (rpid, '0', name, content,like)  #元组
            rpid_list.append(rpid)
            name_list.append(name)
            content_list.append(content)
            like_list.append(like)
            rcount_list.append(rcount)
            # try:
            #     cursor.execute(sql, data)
            #     conn.commit()
            # except:
            #     pass
            print(rpid, ' ', name, ' ', content, ' ', like,' ',rcount)
            # 如果有子评论，爬取子评论
            if node['replies'] is not None:
                print('>>>>>>>>>')
                getSecondReplies(rpid)
        # 每爬完一页，页数加1
        if i%5 == 0:
            print('sleeping....')
            time.sleep(random.uniform(1,5))
        i += 1


if __name__ == '__main__':
    JP = JsonProcess()
    filepath = './data/bilibilicomment/bilibilicomment_2019.csv'
    getReplies(JP, 1)
    df = pd.DataFrame({
        'rpid': rpid_list,
        'name': name_list,
        'content': content_list,
        'like': like_list,
        'rcount': rcount_list,
    })
    header = ['rpid','name','content','like','rcount']
    df.to_csv(filepath,mode='a+',index=False,header=header,encoding='utf-8')
    print('csv saved:{}'.format(filepath))
    print('\n================存储完成================\n')
    # conn.close()

