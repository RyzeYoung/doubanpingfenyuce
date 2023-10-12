import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import csv
import time
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import jieba
from PIL import Image

delay = 5

def get_ciyun_comments():
    #movie_name = input("请输入电影名称：")
    movie_name = "肖申克的救赎"
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

    with open('stopwords.txt', 'a', encoding='utf-8') as file:
        items = jieba.cut(str(movie_name), cut_all=True)
        for item in items:
            file.write(str(item) + '\n')

    comments = []
    for i in range(30):
        response_comment = requests.get(
            "https://movie.douban.com/subject/" + str(sid) + "/comments?start=" + str(
                i * 20) + "&limit=20&status=P&sort=new_score", headers=headers)
        html_comment = response_comment.text

        soup_comment = BeautifulSoup(html_comment, 'lxml')
        all_comment_items = soup_comment.find_all("div", attrs={"class": "comment-item"})
        for item in all_comment_items:
            comment = item.find("span", attrs={"class": "short"}).text.strip()
            comments.append(comment)
        time.sleep(delay)

    with open('词云评论.txt', mode='w', encoding='utf-8') as file:
        for i in range(len(comments)):
            file.write(str(comments[i])+'\n')

def shengcheng_ciyun():
    stopword = open('stopwords.txt', 'r', encoding='utf-8')
    content = stopword.read()
    stopwordlist = content.split('\n')
    clean_word = []
    #print(stopwordlist)
    with open('词云评论.txt', 'r', encoding='utf-8') as file:
        lines = []
        for line in file:
            lines.append(line.strip())
    text = ''.join(lines)
    new_text = jieba.lcut(text)
    for word in new_text:
        if word not in stopwordlist:
            clean_word.append(word)
    clean_word_str=' '.join(clean_word)
    img = Image.open("paidaxingtupian-07979981_1.jpg")
    img_array = np.array(img)
    wordcloud = WordCloud(background_color='white' , mask=img_array,font_path='simsun.ttf',max_words=500,max_font_size=150,relative_scaling=0.6,random_state=50,scale=2)
    wordcloud.generate(clean_word_str)
    image_color = ImageColorGenerator(img_array)
    wordcloud.recolor(color_func=image_color)

    plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


#get_ciyun_comments()
#get_ciyun_comments()
shengcheng_ciyun()
