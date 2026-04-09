# 请安装 OpenAI SDK : pip install openai
# apiKey 获取地址： https://console.bce.baidu.com/qianfan/ais/console/apiKey
# 支持的模型列表： https://cloud.baidu.com/doc/qianfan-docs/s/7m95lyy43

from openai import OpenAI
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key='bce-v3/ALTAK-JPFUCWk8PvUrl9DaNyu6q/a6bfe178ac6bf1574882c2b2d43c69094b6f2e11'
)
response = client.chat.completions.create(
    model="qwen3-235b-a22b-instruct-2507", 
    messages=[
    {
        "role": "user",
        "content": "你是谁"
    }
],
    extra_body={ 
        "stream":False
    }
)
print(response)
# import requests
# import json

# def main():
#     url = "https://qianfan.baidubce.com/v2/models"
    
#     payload = json.dumps({
#     })
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': 'Bearer bce-v3/ALTAK-JPFUCWk8PvUrl9DaNyu6q/a6bfe178ac6bf1574882c2b2d43c69094b6f2e11'
#     }
    
#     response = requests.request("GET", url, headers=headers, data=payload)
    
#     print(response.text)
    

# if __name__ == '__main__':
#     main()