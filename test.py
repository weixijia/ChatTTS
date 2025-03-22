import ChatTTS
import os
import torch
import numpy as np
import wave
import gc
import openai
import pickle
import json
from tools.audio import load_audio

# 初始化 LLM 客户端
llm_api_key = 'llama3.1'
llm_client = openai.OpenAI(api_key=llm_api_key, base_url='http://localhost:11434/v1')

def llm_chat(user_query):
    """与 LLM 对话并获取回复"""
    response = llm_client.chat.completions.create(
        model="llama3.1",
        messages=[
            {"role": "system", "content": """You are a chatbot, reply me always briefly in plain text in Chinese, no non-text content allowed"""},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=1000,  # 减小最大token数以降低内存使用
    )
    return response.choices[0].message.content

def normalize_audio(audio_data, target_level=-23.0):
    """标准化音频电平"""
    if torch.is_tensor(audio_data):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = np.array(audio_data)
    
    # 计算RMS电平
    rms = np.sqrt(np.mean(audio_np**2))
    current_level = 20 * np.log10(rms) if rms > 0 else -100
    
    # 计算需要的增益
    gain = 10**((target_level - current_level) / 20)
    
    # 应用增益
    normalized_audio = audio_np * gain
    
    # 防止削波(clipping)
    if np.max(np.abs(normalized_audio)) > 0.99:
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.99
    
    return normalized_audio

def save_wav(audio_data, sample_rate, file_path):
    """将音频数据保存为 WAV 文件"""
    if torch.is_tensor(audio_data):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = np.array(audio_data)
    
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    
    # 应用音频标准化处理
    audio_np = normalize_audio(audio_np)
        
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"已成功保存音频到: {file_path}")
    return True

def free_memory():
    """释放 CUDA 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("已清理 CUDA 缓存和垃圾回收")

def save_voice_features(spk_smp, transcript, features_path):
    """保存语音特征到文件"""
    # 创建保存目录
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    
    # 将speaker sample特征转换为可序列化的格式
    serializable_spk_smp = None
    if hasattr(spk_smp, "cpu"):
        serializable_spk_smp = spk_smp.cpu().numpy().tolist()
    else:
        # 如果已经是numpy数组或其他可序列化格式
        serializable_spk_smp = spk_smp
    
    # 创建特征数据字典
    feature_data = {
        "spk_smp": serializable_spk_smp,
        "transcript": transcript
    }
    
    # 保存到文件
    with open(features_path, 'wb') as f:
        pickle.dump(feature_data, f)
    
    print(f"语音特征已保存到: {features_path}")
    return True

def load_voice_features(features_path):
    """从文件加载语音特征"""
    if not os.path.exists(features_path):
        print(f"特征文件不存在: {features_path}")
        return None, None
    
    try:
        with open(features_path, 'rb') as f:
            feature_data = pickle.load(f)
        
        spk_smp = feature_data.get("spk_smp")
        transcript = feature_data.get("transcript")
        
        # 如果需要将列表转换回tensor
        if isinstance(spk_smp, list):
            spk_smp = torch.tensor(spk_smp)
            if torch.cuda.is_available():
                spk_smp = spk_smp.cuda()
        
        print(f"成功从 {features_path} 加载语音特征")
        return spk_smp, transcript
    except Exception as e:
        print(f"加载语音特征时出错: {str(e)}")
        return None, None

def smart_text_segmentation(text, max_len=100):
    """智能分段文本，尝试在标点符号处分段"""
    if len(text) <= max_len:
        return [text]
    
    # 中文标点符号和英文标点符号
    punctuations = ["。", "！", "？", "；", "，", ".", "!", "?", ";", ","]
    segments = []
    start = 0
    
    while start < len(text):
        # 如果剩余文本已经小于最大长度，直接添加
        if start + max_len >= len(text):
            segments.append(text[start:])
            break
        
        # 尝试在最大长度内找标点符号
        end = start + max_len
        found = False
        
        # 从最大长度位置向前查找最近的标点符号
        for i in range(end, start, -1):
            if i < len(text) and text[i] in punctuations:
                segments.append(text[start:i+1])
                start = i + 1
                found = True
                break
        
        # 如果没找到标点符号，则按最大长度直接分割
        if not found:
            segments.append(text[start:end])
            start = end
    
    return segments

# 主函数
def main(
        llm_input="请简单介绍一下人工智能技术", 
        sample_path="weixijia_sample.wav",
        max_segment_length=100,
        memory_fraction=0.7,
        features_path="voice_features/sample_features.pkl",
        output_path="output_cloned_voice.wav",
        speech_rate=1.0,           # 语音速率，1.0为正常速度
        noise_reduction=True,      # 是否进行降噪处理
        voice_pitch=0.0,           # 音高调整，0为正常音高
        force_extract=False        # 是否强制重新提取语音特征
    ):
    """
    主函数，可通过参数控制生成行为
    
    参数:
        llm_input (str): 发送给LLM的提示文本
        sample_path (str): 声音样本文件路径
        max_segment_length (int): 文本分段的最大长度
        memory_fraction (float): GPU内存使用比例(0.0-1.0)
        features_path (str): 语音特征保存/加载路径
        output_path (str): 克隆语音输出路径
        speech_rate (float): 语音速率，1.0为正常速度
        noise_reduction (bool): 是否进行降噪处理
        voice_pitch (float): 音高调整，0为正常音高
        force_extract (bool): 是否强制重新提取语音特征
    """
    try:
        # 设置CUDA内存分配策略，减少内存碎片
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # 限制CUDA内存使用量
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        print("初始化 ChatTTS...")
        chat = ChatTTS.Chat()
        
        print("加载模型中...")
        chat.load(compile=False)
        
        # 向 LLM 发送请求
        print("向 LLM 发送请求...")
        say = llm_chat(llm_input)
        print(f"LLM 回复: {say}")
        
        # 加载或提取语音特征
        spk_smp = None
        exact_transcript = """The popularity of wearable and mobile sensing devices have given rise to an increasing interest in complex sensing applications, for example, user activity recognition, but leveraging the existing low-cost sensors, including optical sensors such as RGB cameras or depth RGB or wearable sensors of accelerometer, drop-scope, and we're now able to track a wide range of human behaviors."""
        
        # 如果不强制提取且特征文件存在，则尝试加载
        if not force_extract and os.path.exists(features_path):
            print(f"检测到现有语音特征文件: {features_path}")
            spk_smp, loaded_transcript = load_voice_features(features_path)
            if spk_smp is not None:
                exact_transcript = loaded_transcript or exact_transcript
                print("成功加载语音特征")
            else:
                print("加载语音特征失败，将重新提取")
        
        # 如果没有加载到语音特征，则从音频文件提取
        if spk_smp is None:
            # 使用新提供的样本音频和文本转写
            print("开始从音频中提取语音特征...")
            
            # 检查音频文件是否存在
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"样本音频文件 {sample_path} 不存在")
                
            audio_data = load_audio(sample_path, 24000)
            
            # 提取特征前先释放内存
            free_memory()
            
            # 降低音频处理的批量大小或分段处理
            # 如果样本较长，可以考虑只使用前10秒
            max_len = 24000 * 10  # 10秒的音频
            if len(audio_data) > max_len:
                audio_data = audio_data[:max_len]
            
            spk_smp = chat.sample_audio_speaker(audio_data)
            print("成功提取语音特征")
            
            # 保存语音特征供以后使用
            save_voice_features(spk_smp, exact_transcript, features_path)
        
        # 使用确切的文本转写创建参数
        print("创建声音克隆参数...")
        free_memory()
        
        # 注意：根据 core.py 调整参数，可能有更多自定义选项
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_smp=spk_smp,
            txt_smp=exact_transcript,
            # 以下参数需根据 core.py 的内容确认是否支持
            # speech_rate=speech_rate,            # 语速控制
            # noise_reduction=noise_reduction,    # 降噪控制  
            # voice_pitch=voice_pitch             # 音高控制
        )
        
        # 生成克隆语音
        print("生成克隆语音...")
        free_memory()
        
        # 智能分段处理
        text_segments = smart_text_segmentation(say, max_segment_length)
        
        if len(text_segments) > 1:
            print(f"文本已智能分段为{len(text_segments)}个片段")
            all_cloned_wavs = []
            
            for i, segment in enumerate(text_segments):
                print(f"处理文本段 {i+1}/{len(text_segments)}: {segment[:20]}...")
                
                # 注意：infer方法可能支持更多参数，根据core.py确认
                seg_wav = chat.infer(
                    segment,
                    params_infer_code=params_infer_code
                    # 可能的其他参数
                    # speech_rate=speech_rate,
                    # noise_reduction=noise_reduction,
                    # voice_pitch=voice_pitch
                )
                
                if isinstance(seg_wav, list) and len(seg_wav) > 0:
                    all_cloned_wavs.append(seg_wav[0])
                
                free_memory()  # 每段处理后释放内存
            
            # 合并所有音频段
            if all_cloned_wavs:
                # 添加短暂停顿
                if torch.cuda.is_available():
                    pause = torch.zeros(int(24000 * 0.3)).to(all_cloned_wavs[0].device)  # 0.3秒的停顿
                else:
                    pause = torch.zeros(int(24000 * 0.3))
                
                # 在每个片段之间添加停顿
                combined_segments = []
                for wav in all_cloned_wavs:
                    combined_segments.append(wav)
                    combined_segments.append(pause)
                
                # 去掉最后一个停顿
                combined_segments = combined_segments[:-1]
                
                cloned_wav = torch.cat(combined_segments, dim=0)
                print(f"成功合成所有音频片段，总长度: {len(cloned_wav)/24000:.2f}秒")
            else:
                cloned_wav = None
        else:
            # 注意：infer方法可能支持更多参数，根据core.py确认
            cloned_wav = chat.infer(
                say,
                params_infer_code=params_infer_code
                # 可能的其他参数
                # speech_rate=speech_rate,
                # noise_reduction=noise_reduction,
                # voice_pitch=voice_pitch
            )
            if isinstance(cloned_wav, list) and len(cloned_wav) > 0:
                cloned_wav = cloned_wav[0]
        
        if cloned_wav is not None and len(cloned_wav) > 0:
            save_wav(cloned_wav, 24000, output_path)
            print("成功生成克隆语音!")
        else:
            print("克隆语音生成失败")
                
    except Exception as e:
        print(f"错误: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="ChatTTS 文本到语音转换工具")
    parser.add_argument("--text", type=str, default="请简单介绍一下人工智能技术", help="要转换为语音的文本")
    parser.add_argument("--sample", type=str, default="weixijia_sample.wav", help="声音样本文件路径")
    parser.add_argument("--segment-length", type=int, default=100, help="文本分段的最大长度")
    parser.add_argument("--memory-fraction", type=float, default=0.7, help="GPU内存使用比例(0.0-1.0)")
    parser.add_argument("--features-path", type=str, default="voice_features/sample_features.pkl", help="语音特征保存/加载路径")
    parser.add_argument("--output", type=str, default="output_cloned_voice.wav", help="克隆语音输出路径")
    parser.add_argument("--force-extract", action="store_true", help="强制重新提取语音特征")
    parser.add_argument("--speech-rate", type=float, default=1.0, help="语音速率(0.5-2.0)")
    parser.add_argument("--no-noise-reduction", action="store_false", dest="noise_reduction", help="禁用降噪处理")
    parser.add_argument("--voice-pitch", type=float, default=0.0, help="音高调整(-1.0到1.0)")
    
    args = parser.parse_args()
    
    # 调用主函数
    main(
        llm_input=args.text,
        sample_path=args.sample,
        max_segment_length=args.segment_length,
        memory_fraction=args.memory_fraction,
        features_path=args.features_path,
        output_path=args.output,
        speech_rate=args.speech_rate,
        noise_reduction=args.noise_reduction,
        voice_pitch=args.voice_pitch,
        force_extract=args.force_extract
    )