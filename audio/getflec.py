from moviepy.editor import VideoFileClip
import soundfile as sf

def extract_audio(input_video_path, output_audio_path):
    # 加载视频
    video = VideoFileClip(input_video_path)
    # 提取音频
    audio = video.audio
    # 保存为 FLAC 格式
    audio.write_audiofile(output_audio_path, codec='flac')

if __name__ == '__main__':
    # 示例：提取 MP4 视频中的音频为 FLAC 格式
    input_video = "./final_output_copy.mp4"  # Windows 路径可以使用 /mnt/c/ 在 WSL 中
    output_audio = "./output.flac"
    extract_audio(input_video, output_audio)
