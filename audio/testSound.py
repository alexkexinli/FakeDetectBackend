import soundfile as sf

def process_audio_with_soundfile(file_path):
    # 使用 soundfile 加载音频文件
    audio, sr = sf.read(file_path)
    return audio, sr

if __name__ == '__main__':
    # 测试
    audio_file_path = './test_audio/example.flac'  # 替换为你的 .flac 文件路径
    audio, sr = process_audio_with_soundfile(audio_file_path)
    print(f"Loaded audio with shape: {audio.shape}, Sample rate: {sr}")
