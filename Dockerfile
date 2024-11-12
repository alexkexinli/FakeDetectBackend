FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt



# 暴露端口（FastAPI 默认使用 8000 端口）
EXPOSE 8000

# 启动应用程序
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]