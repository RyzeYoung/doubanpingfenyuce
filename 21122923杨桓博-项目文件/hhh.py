from flask import Flask, render_template
import 词云

app = Flask(__name__)

@app.route('/')
def generate_wordcloud():
    # 调用生成词云的函数
    词云.shengcheng_ciyun()
    # 返回词云结果的HTML模板
    return render_template('wordcloud.html')

if __name__ == '__main__':
    app.run()