import requests

# 替换为实际访问地址
url = 'http://1985783384326394.us-east-1.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20251204_mm0s/v1/chat/completions'
# header信息 Authorization的值为实际的Token
headers = {
    "Content-Type": "application/json",
    "Authorization": "YzIxOTBkMTVkODhmZWFhNzA4YjNmYTRiYTFhY2EzNzY4MDNjMTU5MQ==",
}
# 根据具体模型要求的数据格式构造服务请求。
data = {
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": "You are a math assistant. Solve the problem step by step, explain your reasoning, and box the final answer using \\boxed{}.\n\nFactor $r^2+10r+25$."
    }
  ],
  "max_tokens": 1024
}

# 发送请求
resp = requests.post(url, json=data, headers=headers)
print(resp)
print(resp.content)