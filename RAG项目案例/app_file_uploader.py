import streamlit as st
import pdfplumber
from docx import Document
from io import BytesIO

st.title("知识库更新服务")

#file_uploader
uploader_file = st.file_uploader(
    "上传文件", type=["txt", "pdf", "docx"],
    accept_multiple_files= False,
)

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size /1024 # 转换为KB

    st.subheader(f"文件名:{file_name}")
    st.write(f"文件类型:{file_type}")
    st.write(f"文件大小:{file_size:.2f} KB")

    text = ""
    if file_name.endswith(".pdf"):
        with pdfplumber.open(BytesIO(uploader_file.getvalue())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_name.endswith(".docx"):
        doc = Document(BytesIO(uploader_file.getvalue()))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        # txt 文件:尝试多种编码
        raw = uploader_file.getvalue()
        for encoding in ("utf-8", "gbk", "gb2312"):
            try:
                text = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

    st.write("文件内容预览:", text)
