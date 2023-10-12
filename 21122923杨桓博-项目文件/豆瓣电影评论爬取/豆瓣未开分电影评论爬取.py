import requests
import re
from bs4 import BeautifulSoup
import csv
import time

delay = 10

def get_movie_comments_unknown():
    movie_name = input("请输入电影名称：")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50"
    }
    response = requests.get("https://www.douban.com/search?source=suggest&q=" + str(movie_name), headers=headers)
    html = response.text
    soup = BeautifulSoup(html, "lxml")
    all_uid = soup.findAll("a", attrs={"class": "nbg"})
    answer_1 = re.findall('sid:.*?,', str(all_uid))
    answer_2 = answer_1[0]
    sid = re.sub('\D', '', answer_2)

    comments = []
    for i in range(20):
        response_comment = requests.get(
            "https://movie.douban.com/subject/" + str(sid) + "/comments?&start=" + str(
                i * 20) + "&limit=20&status=P&sort=new_score", headers=headers)
        html_comment = response_comment.text

        soup_comment = BeautifulSoup(html_comment, 'lxml')
        all_comment_items = soup_comment.find_all("div", attrs={"class": "comment-item"})
        for item in all_comment_items:
            comment = item.find("span", attrs={"class": "short"}).text.strip()
            comments.append(comment)
        time.sleep(delay)

    with open('../'+movie_name+'.txt', mode='a', encoding='utf-8', newline='') as file:
        for comment in comments:
            file.write(comment+'\n')

get_movie_comments_unknown()
