import requests
import re
from bs4 import BeautifulSoup
import csv
import time

delay = 10

def map_score(score_text):
    if score_text == "力荐":
        return 10
    elif score_text == "推荐":
        return 8
    elif score_text == "还行":
        return 6
    elif score_text == "较差":
        return 4
    elif score_text == "很差":
        return 2
    else:
        return 0

def get_movie_comments():
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
    scores = []
    for i in range(20):
        response_comment = requests.get(
            "https://movie.douban.com/subject/" + str(sid) + "/comments?percent_type=h&start=" + str(
                i * 20) + "&limit=20&status=P&sort=new_score", headers=headers)
        html_comment = response_comment.text

        soup_comment = BeautifulSoup(html_comment, 'lxml')
        all_comment_items = soup_comment.find_all("div", attrs={"class": "comment-item"})
        for item in all_comment_items:
            comment = item.find("span", attrs={"class": "short"}).text.strip()
            score_element = item.find("span", attrs={"class": re.compile(r"allstar.*rating")})
            score_text = score_element.get("title") if score_element else None
            if score_text:
                score = map_score(score_text)
                comments.append(comment)
                scores.append(score)
        time.sleep(delay)

    with open('../comments_good.csv', mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(comments)):
            writer.writerow([comments[i], scores[i]])

get_movie_comments()
