FROM animcogn/face_recognition:cpu

COPY . /root/app
WORKDIR /root/app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt &&  \
    pip3 install -r requirements2.txt
CMD ["python3", "main.py"]