import random
import requests
import re
from bs4 import BeautifulSoup
import csv
import time


def get_movie_review():
    # movie_name = input("请输入电影名称：")
    movie_name = "肖申克的救赎"
    headers = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50" }
    response = requests.get("https://www.douban.com/search?source=suggest&q="+str(movie_name),headers = headers)
    html = response.text
    #print(response.text)
    soup = BeautifulSoup(html, "lxml")
    all_uid = soup.findAll("a", attrs={"class": "nbg"})
    #print(all_uid)
    answer_1 = re.findall('sid:.*?,',str(all_uid))
    answer_2 = answer_1[0]
    sid = re.sub('\D','',answer_2)

    yingping =[]
    response_comment = requests.get("https://movie.douban.com/subject/"+str(sid)+"/reviews", headers = headers)
    html_comment = response_comment.text
    soup_comment = BeautifulSoup(html_comment,'lxml')
    movie_review = soup_comment.findAll("h2")
    #print(movie_review)
    links = []
    for link in movie_review:
        href = re.findall('<a href="(.*?)">',str(link))
        links.append(href)
    address = str(links[1]).strip("[']")
    response_review_full = requests.get(str(address), headers = headers)
    html_comment_full = response_review_full.text
    soup_comment_full = BeautifulSoup(html_comment_full, 'lxml')
    movie_review_full = soup_comment_full.findAll("div",attrs={"class":"review-content clearfix"})
    #print(movie_review_full)
    review_full = []
    comp = re.compile('<div.*?>(.*)</div>', re.S)
    review_full.append(comp.findall(str(''.join(re.split('[\n ]', str(movie_review_full))))))
    review_clean = str(review_full).replace("<br/>","").strip("[']")
    #qprint(review_clean)
    with open('网站/static/yingping_good肖申克的救赎.txt', mode='a', encoding='utf-8', newline='') as file:
            file.write(review_clean)
    with open('D:\wangzhan\网站\static\yingping_hao_wangye'+str(movie_name)+'.txt', mode='a', encoding='utf-8', newline='') as file:
            file.write(review_clean)
    #print(yingping)

# 测试函数
get_movie_review()