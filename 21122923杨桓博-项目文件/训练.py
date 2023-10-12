import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import jieba

# 读取CSV文件
data = pd.read_csv('comments_good.csv', header=None, encoding='utf-8')

# 获取评论和得分数据
reviews = data.iloc[:, 0].fillna('').tolist()
scores = data.iloc[:, 1].fillna(0).tolist()

# 分词
segmented_reviews = [' '.join(jieba.lcut(review)) for review in reviews]

# 实例化词袋模型
vectorizer = CountVectorizer()

# 将分词后的评论文本转换为特征向量
X = vectorizer.fit_transform(segmented_reviews)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)

# 转换稀疏矩阵为NumPy数组
X_train = X_train.toarray()
X_test = X_test.toarray()

# 转换y_train和y_test为NumPy数组
y_train = np.array(y_train)
y_test = np.array(y_test)

# 构建线性回归模型
model = tf.keras.Sequential([
    layers.Dense(1, activation='linear', input_shape=(X_train.shape[1],))
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# 模型训练
model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test))
model.save('linear_regression_model.h5')

# 模型评估
loss, mae, mse = model.evaluate(X_test, y_test)
print("模型均方误差：", mse)

loaded_model = tf.keras.models.load_model('linear_regression_model.h5')

# 对新的评论进行预测
new_reviews = ['烂片',
               '从前的爱情，青涩懵懂，电影和上海渊源深厚，却最多只具体到一行滑落的地址',
               '《云在江口》电影可以帮助观众舒缓心情，透过银幕进入山水之间，从而鼓励人们走出去，体会不一样的生活之美']
new_segmented_reviews = [' '.join(jieba.lcut(review)) for review in new_reviews]
new_X = vectorizer.transform(new_segmented_reviews).toarray()
predictions = model.predict(new_X)
predictions = np.clip(predictions, 0, 10)
print("新评论的预测得分：", predictions)
