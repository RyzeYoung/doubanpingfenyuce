from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import json
import re
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import jieba
from PIL import Image
import numpy as np
from snownlp import SnowNLP
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from jieba import analyse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from langdetect import detect
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from textrank4zh import TextRank4Sentence

app = Flask(__name__)

data = pd.read_csv('D:\wangzhan\comments_good.csv', header=None, encoding='utf-8')
reviews = data.iloc[:, 0].fillna('').tolist()
segmented_reviews = [' '.join(jieba.lcut(review)) for review in reviews]
vectorizer = CountVectorizer()
vectorizer.fit_transform(segmented_reviews)
model = tf.keras.models.load_model('D:\wangzhan\linear_regression_model.h5')



def extract_keywords_english(text, top_k):
    # 分词
    tokens = word_tokenize(text.lower())
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    # 构建语料库
    corpus = [' '.join(filtered_tokens)]
    # 计算TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # 获取特征词和对应的TF-IDF值
    feature_names = vectorizer.get_feature_names()
    tfidf_scores = tfidf_matrix.toarray()[0]
    # 提取前top_k个关键词
    keywords = [feature_names[index] for index in tfidf_scores.argsort()[::-1][:top_k]]
    return keywords




def generate_wordcloud():
    stopword = open('D:\wangzhan\stopwords.txt', 'r', encoding='utf-8')
    content = stopword.read()
    stopwordlist = content.split('\n')
    clean_word = []
    #print(stopwordlist)
    with open('D:\wangzhan\词云评论.txt', 'r', encoding='utf-8') as file:
        lines = []
        for line in file:
            lines.append(line.strip())
    text = ''.join(lines)
    new_text = jieba.lcut(text)
    for word in new_text:
        if word not in stopwordlist:
            clean_word.append(word)
    clean_word_str=' '.join(clean_word)
    img = Image.open("D:\wangzhan\paidaxingtupian-07979981_1.jpg")
    img_array = np.array(img)
    wordcloud = WordCloud(background_color='white' , mask=img_array,font_path='D:\wangzhan\simsun.ttf',max_words=500,max_font_size=150,relative_scaling=0.6,random_state=50,scale=2)
    wordcloud.generate(clean_word_str)
    image_color = ImageColorGenerator(img_array)
    wordcloud.recolor(color_func=image_color)



def summarize_text_chinese(movie_name):
    file_path = r'D:\wangzhan\网站\static\yingping_good' + str(movie_name) + '.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=content, lower=True, source='all_filters')
    summary_sentences = tr4s.get_key_sentences(num=3, sentence_min_len=5)
    summary = ' '.join([sentence.sentence for sentence in summary_sentences])

    return summary


def summarize_text_chinese_cha(movie_name):
    file_path = r'D:\wangzhan\网站\static\yingping_bad' + str(movie_name) + '.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=content, lower=True, source='all_filters')
    summary_sentences = tr4s.get_key_sentences(num=3, sentence_min_len=5)
    summary = ' '.join([sentence.sentence for sentence in summary_sentences])

    return summary



def analyze_sentiment(sentence):
    s = SnowNLP(sentence)
    sentiment = s.sentiments
    if sentiment > 0.5:
        return "Positive"
    elif sentiment < 0.5:
        return "Negative"
    else:
        return "Neutral"




@app.route('/analysis/<movie_name>')
def analysis(movie_name):
    summary = summarize_text_chinese(movie_name)  # 调用路由获取数据
    summary_cha = summarize_text_chinese_cha(movie_name)
    generate_wordcloud()
    image_path = movie_name+'.png'
    with open('D:/wangzhan/网站/static/yingping_hao_wangye'+str(movie_name)+'.txt', 'r',encoding='utf-8') as file:
        content = file.read()
    with open('D:/wangzhan/网站/static/yingping_cha_wangye'+str(movie_name)+'.txt', 'r',encoding='utf-8') as file:
        content_cha = file.read()
    image_url = movie_name+".jpg"
    with open('D:/wangzhan/网站/static/比站视频推荐链接'+str(movie_name)+'.txt', 'r',encoding='utf-8') as file:
        url_bilibili = file.read()
    return render_template('analysis.html', summary=summary,content=content,summary_cha=summary_cha,content_cha=content_cha,image_path=image_path,movie_name=movie_name,image_url=image_url,url_bilibili=url_bilibili)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name_analysis = request.form.get('movie_name_analysis')  # 获取分析页面的电影名称
        movie_name_prediction = request.form.get('movie_name_prediction')  # 获取预测页面的电影名称

        if movie_name_analysis:
            return redirect(url_for('analysis', movie_name=movie_name_analysis))  # 跳转到分析页面

        if movie_name_prediction:
            return redirect(url_for('prediction', movie_name=movie_name_prediction))  # 跳转到预测页面

    return render_template('index.html')

@app.route('/back_to_home', methods=['GET'])
def back_to_home():
    return redirect(url_for('index'))

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    sentence = data['sentence']
    sentiment = perform_sentiment_analysis(sentence)
    return jsonify({'sentiment': sentiment})

def perform_sentiment_analysis(sentence):
        s = SnowNLP(sentence)
        sentiment = s.sentiments
        if sentiment > 0.8:
            return "Positive"
        elif sentiment < 0.4:
            return "Negative"
        else:
            return "Neutral"


@app.route('/prediction/<movie_name>')
def prediction(movie_name):
    with open('D:/wangzhan/网站/static/'+str(movie_name)+'.txt', 'r', encoding='utf-8') as file:
        new_reviews = file.readlines()[:10]  # 使用用户输入的电影名称作为新评论，并截取前十条评论
    new_segmented_reviews = [' '.join(jieba.lcut(review)) for review in new_reviews]
    new_X = vectorizer.transform(new_segmented_reviews).toarray()
    predictions = model.predict(new_X)
    predictions = np.clip(predictions, 0, 10)
    average_score = np.mean(predictions)

    # 对每条评论进行情感分析，并将结果添加到评论后面
    sentiment_results = []
    for review in new_reviews:
        result = perform_sentiment_analysis(review)
        sentiment_results.append(result)

    with open('D:/wangzhan/网站/static/'+str(movie_name)+'简介.txt','r',encoding='utf-8') as file1:
        jianjie = file1.read()
    language = detect(jianjie)
    if language == 'en':
        keywords = extract_keywords_english(jianjie, top_k=3)
    elif language == 'zh-cn':
        keywords = analyse.textrank(jianjie, topK=3)
    else:
        keywords = extract_keywords_english(jianjie, top_k=3)

    return render_template('prediction.html', movie_name=movie_name, average_score=average_score, reviews=new_reviews, jianjie=jianjie, keywords=keywords, sentiment_results=sentiment_results)



if __name__ == '__main__':
    app.run()
