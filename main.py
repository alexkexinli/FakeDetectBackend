from fastapi import FastAPI, File, UploadFile
from moviepy.editor import VideoFileClip
from io import BytesIO
import numpy as np
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
import random
import string
from io import BytesIO
from audio.getflec import extract_audio
from audio.test import evaluate_audio_from_path
app = FastAPI()
# 定义保存目录
UPLOAD_DIR = Path("uploads/")
FRAME_DIR = Path("frames/")
UPLOAD_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)

model_path = 'model_epoch_last_2.pth'
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
    return realcounter >=8,y_pred
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 检查文件类型
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a video")

    # file_bytes = await file.read()
    # file_stream = BytesIO(file_bytes)
    #
    # file_array = np.frombuffer(file_stream.read(), np.uint8)
    # cap = cv2.VideoCapture(cv2.imdecode(file_array, cv2.IMREAD_COLOR))
    random_suffix=generate_random_suffix()
    # 保存视频文件
    video_path = UPLOAD_DIR / (random_suffix+file.filename)
    audio_path=UPLOAD_DIR / (random_suffix+".flac")
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extract_audio(str(video_path),str(audio_path))
    aud_rst,aud_pos=evaluate_audio_from_path(str(audio_path))

    # 读取视频文件
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    total_frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"fps is {fps},frame count is {total_frame_count}")
    interval = max(int(fps / 2), 1)  # 计算每秒截取2帧的间隔

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
                top, right, bottom, left = face_locations[0]
                # 扩展裁剪区域，保留更多面部特征（可根据需要调整）
                h, w, _ = rgb_frame.shape
                padding_w = int(0.4 * (right - left))
                padding_h = int(0.4 * (bottom - top))
                left = max(left - padding_w, 0)
                top = max(top - padding_h, 0)
                right = min(right + padding_w, w)
                bottom = min(bottom + padding_h, h)
                cropped_img = rgb_frame[top:bottom, left:right]
                face_frames.append(cropped_img)
                # 如果已达到10张图片，停止处理
                if len(face_frames) >= 10:
                    break
        frame_count += 1

    cap.release()

    if not face_frames:
        return JSONResponse(content={"message": "No faces detected in the video."})



    # 将结果转换为列表
    video_rst,vid_pos = predict(face_frames)

    rst=video_rst&aud_rst
    return {"result": rst,"aud_pos":aud_pos,"vid_pos":vid_pos }


@app.post("/detect2")
async def detect2(file: UploadFile = File(...)):
    # 检查文件类型
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a video")

    video_path = UPLOAD_DIR / (generate_random_suffix()+file.filename)
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # 将上传文件读取为字节数据


    # 使用 moviepy 从字节流中读取视频
    clip = VideoFileClip(str(video_path))
    # 逐帧读取并处理视频
    frames = []
    interval = 0.5  # 每秒提取 2 帧
    t = 0
    while t < clip.duration:
        frame = clip.get_frame(t)  # 获取在时间 t 处的视频帧
        frames.append(frame)  # 保存每一帧（以 NumPy 数组格式保存）
        t += interval  # 每次增加 0.5 秒

    # 释放资源
    clip.reader.close()

    print("size "+ len(frames))
    if not frames:
        return JSONResponse(content={"message": "No faces detected in the video."})



    # 将结果转换为列表
    predictions = predict(frames)

    return {"result": predictions }

def generate_random_suffix(length: int = 8) -> str:
    """生成随机字符串后缀"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
