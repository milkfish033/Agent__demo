"""
知识库服务基础代码
"""
import os

from dotenv import load_dotenv 
import config_data as config
import hashlib 
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

load_dotenv()
#检查文件是被处理过了
def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        #文件不存在，没处理过
        open(config.md5_path, "w", encoding="utf-8").close() #创建空文件
        return False
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False
    




#将传入的md5文件保存到数据库中
def save_md5(md5_str: str):
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")





#将传入的字符串改为md5字符串
def get_string_md5(input_str: str, encoding = "utf-8"):
    #将字符串转化为bytes字节数组
    byte_str = input_str.encode(encoding)
    #创建md5对象
    md5_obj = hashlib.md5()
    #更新md5对象
    md5_obj.update(byte_str)
    #获取md5字符串
    md5_str = md5_obj.hexdigest()
    return md5_str





class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True) #创建数据库本地存储文件夹
        
        self.chroma = Chroma(
            collection_name = config.collection_name, #数据库的表名
            embedding_function= DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory= config.persist_directory #数据库本地存储文件夹
        ) #向量存储的chroma实例对象


        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,
            chunk_overlap= config.chunk_overlap,
            separators= config.separators,
            length_function = len
        ) #文本切分器实例对象



    def upload_by_str(self, data, filename):
        """
        将传入的字符串进行向量化， 并保存到向量数据库中
        data: 需要上传的数据字符串
        param filename: 文件名,用于生成md5
        """
        md5_hex = get_string_md5(data) #获取数据字符串的md5值
        if check_md5(md5_hex):
            print(f"{filename} has been processed before, skip it.")
            return
        
        #文本切分
        if len(data) > config.max_spilter_char_number:
            knowledge = self.spliter.split_text(data)

        else:
            knowledge = [data]

        metadata = {
            "source": filename, 
            "create_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "admin"
            }
        
        self.chroma.add_texts(knowledge, metadatas=[metadata for _ in knowledge]) #向量化并保存到数据库中
        save_md5(md5_hex) #保存md5值到文件中
        return "Upload success."
