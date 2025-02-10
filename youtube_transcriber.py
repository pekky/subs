import yt_dlp
import whisperx
import torch
import srt
from datetime import timedelta
import os

def download_audio(url, output_path="audio.mp3"):
    """从 YouTube 下载音频"""
    print("正在下载音频...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.mp3', ''),
        'quiet': True,
        'no_warnings': True,
        'cookies': 'cookies.txt'
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("音频下载完成！")
    except Exception as e:
        raise Exception(f"下载失败：{str(e)}")

def transcribe_audio(audio_path):
    """使用 WhisperX 进行转录和说话人分离"""
    print("正在进行语音识别和说话人分离...")
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = whisperx.load_model("large-v2", device)
    
    # 转录音频
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    print("语音识别完成！")
    
    # 加载说话人分离模型
    print("正在进行说话人分离...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(audio)
    
    # 将说话人信息添加到转录结果中
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print("说话人分离完成！")
    
    return result

def create_srt(transcription, output_path="output.srt"):
    """生成 SRT 字幕文件"""
    srt_segments = []
    for i, segment in enumerate(transcription["segments"], 1):
        # 获取时间戳
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        
        # 获取说话人标识
        speaker = f"说话人 {segment.get('speaker', 'UNKNOWN')}"
        
        # 创建字幕文本
        text = f"{speaker}：{segment['text']}"
        
        # 创建 SRT 字幕项
        subtitle = srt.Subtitle(index=i, 
                              start=start_time, 
                              end=end_time, 
                              content=text)
        srt_segments.append(subtitle)
    
    # 写入 SRT 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(srt_segments))

def process_youtube_video(url):
    """处理 YouTube 视频的主函数"""
    try:
        # 下载音频
        audio_path = "audio.mp3"
        download_audio(url, audio_path)
        
        # 转录并进行说话人分离
        transcription = transcribe_audio(audio_path)
        
        # 生成 SRT 文件
        print("正在生成字幕文件...")
        create_srt(transcription)
        
        # 清理临时文件
        os.remove(audio_path)
        
        print("\n✨ 处理完成！字幕文件已保存为 output.srt")
    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    print("YouTube 视频转录工具")
    print("-------------------")
    while True:
        try:
            youtube_url = input("\n请输入 YouTube 视频 URL（输入 q 退出）：").strip()
            if youtube_url.lower() == 'q':
                print("感谢使用！")
                break
            if not youtube_url:
                print("URL 不能为空！")
                continue
                
            process_youtube_video(youtube_url)
            
        except KeyboardInterrupt:
            print("\n\n程序已终止")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{str(e)}") 