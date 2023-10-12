import requests
import json
import re

def main():
    with open('D:\wangzhan\yingping_good.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    movie_comment = content
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/news_summary?charset=UTF-8&access_token=" + get_access_token()

    payload = json.dumps({
        "content": str(movie_comment),
        "max_summary_len": 100
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    sentence = response.text
    pattern = re.compile(r'"summary":\s*"(.*)"')
    match = pattern.search(str(sentence))

    if match:
        summary = match.group(1)
        print(summary)
    else:
        print("未找到summary字段")


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    API_KEY = "a6MaAMmDgLr1ovSsC5G7CF7M"
    SECRET_KEY = "6UOExkS2VRUmGzfPl7WoEPLGxbn5602t"
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
