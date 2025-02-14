import yt_dlp
import whisperx
import torch
import srt
from datetime import timedelta
import os

def download_audio(mp3_path):
    """从指定路径加载 MP3 音频"""
    print("正在加载音频...")
    
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"音频文件未找到：{mp3_path}")
    
    # 这里可以添加任何需要的处理逻辑
    print(f"成功加载音频文件：{mp3_path}")
    
    return mp3_path  # 返回音频文件路径

def transcribe_audio(audio_path):
    """使用 WhisperX 进行转录和说话人分离"""
    print("正在进行语音识别和说话人分离...")
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = whisperx.load_model("large-v2", device, compute_type="float32")
    
    # 转录音频
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    print("语音识别完成！")
    
    # 加载说话人分离模型
    print("正在进行说话人分离...")
    # 获取 Hugging Face 访问令牌
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=huggingface_token, device=device)
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

def process_audio_file(mp3_path):
    """处理音频文件的主函数"""
    try:
        # 加载音频
        audio_path = download_audio(mp3_path)
        
        # 转录并进行说话人分离
        transcription = transcribe_audio(audio_path)
        
        # 生成 SRT 文件
        print("正在生成字幕文件...")
        create_srt(transcription)
        
        print("\n✨ 处理完成！字幕文件已保存为 output.srt")
    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")

if __name__ == "__main__":
    print("音频转录工具")
    print("-------------------")
    while True:
        try:
            mp3_path = input("\n请输入 MP3 文件路径（输入 q 退出）：").strip()
            if mp3_path.lower() == 'q':
                print("感谢使用！")
                break
            if not mp3_path:
                print("路径不能为空！")
                continue
                
            process_audio_file(mp3_path)
            
        except KeyboardInterrupt:
            print("\n\n程序已终止")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{str(e)}") 