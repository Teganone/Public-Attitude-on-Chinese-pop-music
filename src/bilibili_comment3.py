# 发送请求
import requests
# 将数据存入数据库
# import MySQLdb
# 每次请求停1s，太快会被B站拦截。
import time
import pandas as pd
import threading
# # 连接数据库
# conn = MySQLdb.connect(host="localhost", user='root', password='admin', db='scholldatabase', charset='utf8')
# cursor = conn.cursor()
# # 预编译语句
# sql = "insert into bilibili(rpid,root,name,avatar,content) values (%s,%s,%s,%s,%s)"

# rpid_list = []
# name_list = []
# content_list = []
# like_list = []
# rcount_list = []

# 爬虫类（面向对象）
class JsonProcess:
    def __init__(self):
        self.Json_data = ''
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
	
    # 发送爬取请求
    def spider(self, URL):
        url = URL
        response = requests.get(url, headers=self.headers, verify=False)
        response.encoding = 'utf-8'
        self.Json_data = response.json()['data']['replies']


# 爬取子评论
def getSecondReplies(root,rpid_list,name_list,content_list,like_list,rcount_list):
    reply = JsonProcess()
    # 页数
    pn = 1
    # 不知道具体有多少页的评论，所以使用死循环一直爬
    while pn<=2:
        url = f'https://api.bilibili.com/x/v2/reply/reply?jsonp=jsonp&pn={pn}&type=1&oid=979849123&ps=10&root={root}&_=1647581648753'
        # 没爬一次就睡1秒
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
def getReplies(pnstart,pnend,filepath):
    jp = JsonProcess()
    pn = pnstart
    rpid_list = []
    name_list = []
    content_list = []
    like_list = []
    rcount_list = []

    # 不知道具体有多少页的评论，所以使用死循环一直爬
    while pn<=pnend:
        # url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={pn}&oid=755512750&plat=1&seek_rpid=&type=1'  #2020
        # url = 'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next=0&oid=550272309&plat=1&seek_rpid=&type=1'  #2021
        url = f'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next={pn}&oid=71541567&plat=1&seek_rpid=&type=1'   #2006
        print(url)
        # url = 'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next=2&oid=71541567&plat=1&seek_rpid=&type=1'   #2006
        # url = 'https://api.bilibili.com/x/v2/reply/main?csrf=c70c1107e758668c6f40e565a05d07ce&mode=3&next=0&oid=56920510&plat=1&seek_rpid=&type=1'  #2005
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
                getSecondReplies(rpid,rpid_list,name_list,content_list,like_list,rcount_list)
            # df['rpid'] = rpid_list
            # df['name'] = name_list
            # df['content'] = content_list
            # df['like'] = like_list
            # df['rcount'] = rcount_list
            # df.to_csv(filepath,mode='a+',index=False,encoding='utf-8')
    
            # df.to_csv(filepath,mode='a+',index=False,header=header,encoding='utf-8')
            # rpid_list.clear()
            # name_list.clear()
            # content_list.clear()
            # like_list.clear()
            # rcount_list.clear()
                # 每爬完一页，页数加1
        pn += 1
    # df = pd.DataFrame({
    #     'rpid': rpid_list,
    #     'name': name_list,
    #     'content': content_list,
    #     'like': like_list,
    #     'rcount': rcount_list,
    # })

    
    df = pd.DataFrame()

    # header = ['rpid','name','content','like','rcount']
    df['rpid'] = rpid_list
    df['name'] = name_list
    df['content'] = content_list
    df['like'] = like_list
    df['rcount'] = rcount_list
    if lock.acquire():
        df.to_csv(filepath,mode='a+',index=False,encoding='utf-8')
        lock.release()
    # df.to_csv(filepath,mode='a+',index=False,header=header,encoding='utf-8')

    print('csv saved:{}'.format(filepath))
    print('\n================存储完成================\n')



if __name__ == '__main__':
    # JP = JsonProcess()

    lock = threading.Lock()

    url = []
    year_list = [2005,2006,2012,2013,2020,2021]
    # for year in year_list:
    #     filepath = f'./data/bilibilicomment/bilibilicomment_{year}.csv'
    #     # print(filepath)
    #     # file = open(filepath,'a')
    #     getReplies(1,filepath)
    year = 2006
    filepath = f'./data/bilibilicomment/bilibilicomment_{year}.csv'
    thread1 = threading.Thread(target=getReplies,args=(1,2,filepath))
    thread2 = threading.Thread(target=getReplies,args=(3,4,filepath))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # conn.close()

