import requests
import re
from bs4 import BeautifulSoup


def get_bilibili_reference():
    # movie_name = input("请输入电影名称：")
    movie_name = "肖申克的救赎"
    headers = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50" }
    response = requests.get("https://search.bilibili.com/all?keyword="+str(movie_name),headers = headers)
    html = response.text
    #print(response.text)
    soup = BeautifulSoup(html, "lxml")
    movie_review = soup.findAll("a")
    picture_found = soup.findAll("source")
    #print(movie_review)
    links = []
    new_list = []
    movie_pic = []
    for link in movie_review:
        href = re.findall('<a.*?href="(.*?)".*?>',str(link))
        #print(href)
        links.append(href)
    for item in links:
        if item !=[]:
            new_list.append(item)
    #print(new_list)
    address = str(new_list[0]).strip("[']")
    for pic in picture_found:
        movie_address = re.findall('<source srcset="(.*?)".*?>',str(pic))
        movie_pic.append(movie_address)
    #    print(movie_address)
    download_url = "http:"+str(movie_pic[0][0])
    save_path = 'D:/wangzhan/网站/static/bilibili_tuijian'+str(movie_name)+'.png'
    response = requests.get(download_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    with open("D:/wangzhan/网站/static/比站视频推荐链接"+str(movie_name)+'.txt','w',encoding='utf-8') as file:
        file.write(address)



get_bilibili_reference()