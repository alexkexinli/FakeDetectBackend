from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import shutil
from pathlib import Path
import face_recognition
import torch
from torchvision import transforms
from networks.resnet import resnet50
import numpy as np


app = FastAPI()
# 定义保存目录
UPLOAD_DIR = Path("uploads/")
FRAME_DIR = Path("frames/")
UPLOAD_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)

# 加载你的二分类模型
# 请替换为你的模型路径和加载方式
model_path = 'model_epoch_last.pth'
# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

loadSize=256
crop_func = transforms.Lambda(lambda img: img)
flip_func = transforms.Lambda(lambda img: img)
rz_func = transforms.Resize((loadSize, loadSize))

# 定义图像转换
covert_tensor = transforms.Compose([
        rz_func,
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def predict(imgs):
    imgs = [covert_tensor(img) for img in imgs]
    with torch.no_grad():
        y_pred = []
        for img in imgs:
            print(type(img),"img")
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())

    realcounter = 0
    for pred in y_pred:
        if pred <=0.5:
            realcounter+=1
    return realcounter >=8
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 检查文件类型
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a video")

    # 保存视频文件
    video_path = UPLOAD_DIR / file.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 读取视频文件
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    interval = max(int(fps / 5), 1)  # 计算每秒截取5帧的间隔

    frame_count = 0
    face_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 每隔 interval 帧处理一次
        if frame_count % interval == 0:
            # 将 BGR 转换为 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 检测人脸
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            if face_locations:
            # if True:
                face_frames.append(rgb_frame)
                # 如果已达到10张图片，停止处理
                if len(face_frames) >= 10:
                    break
        frame_count += 1

    cap.release()

    if not face_frames:
        return JSONResponse(content={"message": "No faces detected in the video."})



    # 将结果转换为列表
    predictions = predict(face_frames)

    return {"result": predictions }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)