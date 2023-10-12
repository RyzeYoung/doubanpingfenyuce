import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import jieba

data = pd.read_csv('comments_good.csv', header=None, encoding='utf-8')

# 获取评论和得分数据
reviews = data.iloc[:, 0].fillna('').tolist()

# 分词
segmented_reviews = [' '.join(jieba.lcut(review)) for review in reviews]
vectorizer = CountVectorizer()
vectorizer.fit_transform(segmented_reviews)

model = tf.keras.models.load_model('linear_regression_model.h5')

def predict_sentiment(new_reviews):
    new_segmented_reviews = [' '.join(jieba.lcut(review)) for review in new_reviews]
    new_X = vectorizer.transform(new_segmented_reviews).toarray()
    predictions = model.predict(new_X)
    predictions = np.clip(predictions, 0, 10)
    return predictions

new_reviews = ['烂片，垃圾',
               '从前的爱情，青涩懵懂，电影和上海渊源深厚，却最多只具体到一行滑落的地址',
               '《云在江口》电影可以帮助观众舒缓心情，透过银幕进入山水之间，从而鼓励人们走出去，体会不一样的生活之美']
predictions = predict_sentiment(new_reviews)
average_score = np.mean(predictions)
print("新评论的预测平均得分：", average_score)
