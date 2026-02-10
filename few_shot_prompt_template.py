from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import os

model = ChatTongyi(model="qwen3-max", api_key=os.getenv("DASHSCOPE_API_KEY"))   

exmaple_template = PromptTemplate.from_template("中文: {question},  英文: {answer}")

exmaples = [ 
    {"question": "你是谁？", "answer": "who are you?"},
    {"question": "今天天气怎么样？", "answer": "hows the weather today?"},
]

few_shot_template = FewShotPromptTemplate(
    example_prompt = exmaple_template,
    examples = exmaples,
    prefix = "请将以下中文翻译成英文：",
    suffix = "请翻译：{input_word}",
    input_variables = ["input_word"],
)

res = few_shot_template.invoke(input = {"input_word": "我喜欢编程。"})

print(model.invoke(res) )


