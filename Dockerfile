# 使用 PyTorch 官方镜像（带 Python 3.11、CUDA、cuDNN）
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置无交互模式，防止安装包时阻塞
ENV DEBIAN_FRONTEND=noninteractive
# 切换到腾讯云的 Ubuntu 镜像源，避免连接失败
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.cloud.tencent.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.cloud.tencent.com/ubuntu/|g' /etc/apt/sources.list

# 安装系统依赖（dlib 需要）
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# 设置国内 pip 源（腾讯云）
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# 复制项目代码
COPY . /app
WORKDIR /app

# 安装 Python 依赖（不包括 torch，因为基础镜像已经自带）
RUN pip install --no-cache-dir -r requirements.txt

# 启动 FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
