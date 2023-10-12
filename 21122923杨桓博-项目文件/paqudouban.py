import random
import requests
import re
from bs4 import BeautifulSoup
import csv
import time

def get_movie_comments():
    movie_name = input("请输入电影名称：")
    headers = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50" }
    response = requests.get("https://www.douban.com/search?source=suggest&q="+str(movie_name),headers = headers)
    html = response.text
    print(response.text)
    soup = BeautifulSoup(html, "lxml")
    all_uid = soup.findAll("a", attrs={"class": "nbg"})
    print(all_uid)
    answer_1 = re.findall('sid:.*?,',str(all_uid))
    answer_2 = answer_1[0]
    sid = re.sub('\D','',answer_2)

    goodcomments = []
    for i in range(1):
        time.sleep(random.randint(3,10))
        response_comment = requests.get("https://movie.douban.com/subject/"+str(sid)+"/comments?percent_type=h&start="+str(i*20)+"&limit=20&status=P&sort=new_score", headers = headers)
        html_comment = response_comment.text

        soup_comment = BeautifulSoup(html_comment,'lxml')
        all_goodcomment = soup_comment.findAll("span", attrs={"class": "short"})

        for comment in all_goodcomment:
            comment_short = comment.text.strip()
            goodcomments.append(comment_short)

    with open('comments.csv', mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for comment_1 in goodcomments:
            writer.writerow([comment_1])

# 测试函数
get_movie_comments()








