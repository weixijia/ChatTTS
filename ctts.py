import ChatTTS
import os
import time
import openai
import torch
import numpy as np
import wave

# 初始化 LLM 客户端
llm_api_key = 'llama3.1'
llm_client = openai.OpenAI(api_key=llm_api_key, base_url='http://localhost:11434/v1')

def llm_chat(user_query):
    """与 LLM 对话并获取回复"""
    response = llm_client.chat.completions.create(
        model="llama3.1",
        messages=[
            {"role": "system", "content": """You are a chatbot, You need to reply me everything I ask you in Chinese"""},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    return response.choices[0].message.content

def save_wav(audio_data, sample_rate, file_path):
    """将音频数据保存为 WAV 文件（不使用 torchaudio）"""
    # 确保是 numpy 数组
    if torch.is_tensor(audio_data):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = np.array(audio_data)
        
    # 确保音频是 float32 类型且范围在 [-1, 1] 之间
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
        
    # 转换为 16 位整数 PCM 格式
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # 创建 wave 文件
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位 = 2字节
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"已成功保存音频到: {file_path}")
    return True

# 主函数
def main():
    try:
        print("初始化 ChatTTS...")
        chat = ChatTTS.Chat()
        print("加载模型中...")
        chat.load(compile=False)  # 设置为 True 可提高性能
        
        print("向 LLM 发送请求...")
        say = llm_chat("Hello")
        print(f"LLM 回复: {say}")
        
        print("生成语音中...")
        texts = [say]
        wavs = chat.infer(texts)
        
        if len(wavs) > 0 and wavs[0] is not None:
            print("成功生成语音，准备保存...")
            # 使用自定义函数保存音频，避免 torchaudio 的问题
            save_wav(wavs[0], 24000, "output1.wav")
            print("处理完成!")
        else:
            print("错误: 生成的音频为空")
    
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()