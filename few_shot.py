from openai import OpenAI
import os 
import json 

client = OpenAI(
    api_key= os.getenv("OEPNAI_API_KEY"),
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

data = [
    {
  "text": "我要一扇 ROW100 窗户，宽 120cm，高 150cm，大概多少钱？",
  "entities": {
    "product_model": "ROW100",
    "width_cm": 120,
    "height_cm": 150,
    "intent": "price_inquiry"
  }
},
{
  "text": "在上海浦东的公寓想装隔音窗，大概下个月安装。",
  "entities": {
    "location": "上海浦东",
    "product_feature": "隔音",
    "installation_time": "下个月",
    "intent": "installation_consultation"
  }
},
{
  "text": "我要三扇平开窗，每扇宽 1 米，高 1.4 米。",
  "entities": {
    "window_type": "平开窗",
    "quantity": 3,
    "width_m": 1.0,
    "height_m": 1.4,
    "intent": "product_configuration"
  }
},
{
  "text": "有没有适合阳光房的门窗推荐？",
  "entities": {
    "application_scene": "阳光房",
    "intent": "product_recommendation"
  }
},
{
  "text": "高度是 2 米，宽度还没量。",
  "entities": {
    "height_m": 2.0,
    "width_m": "null",
    "intent": "supplement_information"
  }
}
]

example_data = []
for d in data:
    example_data.append(
        {"role": "user", "content" : d["text"]})
    
    example_data.append(
        {"role" : "assistant", "content": json.dumps(d["entities"], ensure_ascii=False)}
    )
    



questions = [
    "我想在波士顿装三个窗户",
    "海边隔音效果好的窗户推荐"
]

for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages= example_data + [{"role": "user", "content" : f"按照上述实例，抽取这个句子的信息{q}"}]
    )

    print(response.choices[0].message.content)