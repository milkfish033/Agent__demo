md5_path = "./md5.txt" #保存md5的文件路径


#Chroma
collection_name = "rag"
persist_directory= "./chroma_db" #向量数据库保存路径

#spilter 

chunk_size = 1000 #文本切分的块大小
chunk_overlap = 100 #文本切分的重叠部分大小
separators=["\n\n", "\n", "。", "！", "？", ",", "，", " ", "!"] #文本切分的分隔符列表

max_spilter_char_number = 1000