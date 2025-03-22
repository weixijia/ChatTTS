import os
import sys
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QComboBox, QFileDialog, 
                             QTabWidget, QLineEdit, QScrollArea, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QTextCursor
import logging
import openai

# 设置PyTorch内存管理选项
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# PyTorch配置
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

# 导入ChatTTS相关模块
import ChatTTS
import torchaudio
from tools.logger import get_logger
from tools.audio import load_audio

# 检查PyAudio是否可用
try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio playback will be disabled.")

# 设置日志
logger = get_logger("ChatTTS_GUI")

class WorkerThread(QThread):
    """工作线程，用于处理可能阻塞UI的长时间运行任务"""
    
    # 信号定义
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    audio_ready = pyqtSignal(object)
    llm_response = pyqtSignal(str)
    
    def __init__(self, task_type, params=None):
        """
        初始化工作线程
        
        参数:
            task_type: 任务类型 ('load_model', 'generate_audio', 'llm_chat', etc.)
            params: 任务参数
        """
        super().__init__()
        self.task_type = task_type
        self.params = params or {}
        self.chat = None
        self.llm_client = None
        self.spk_smp = None
        
    def run(self):
        """运行指定任务"""
        try:
            if self.task_type == 'load_model':
                self._load_model()
            elif self.task_type == 'extract_speaker':
                self._extract_speaker()
            elif self.task_type == 'generate_audio':
                self._generate_audio()
            elif self.task_type == 'llm_chat':
                self._llm_chat()
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}")
            self.error.emit(f"错误: {str(e)}")
            
    def _load_model(self):
        """加载ChatTTS模型"""
        self.progress.emit("正在加载ChatTTS模型...")
        
        # 选择设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress.emit(f"使用设备: {device}")
        
        # 初始化ChatTTS
        self.chat = ChatTTS.Chat()
        
        # 加载模型 (使用与修改后代码一致的方法)
        self.chat.load(compile=False)
        self.progress.emit("ChatTTS模型加载完成")
        
        # 初始化LLM客户端
        try:
            self.llm_client = openai.OpenAI(
                api_key="llama3.1", 
                base_url="http://localhost:11434/v1"
            )
            self.progress.emit("已初始化LLM客户端")
        except Exception as e:
            self.progress.emit(f"初始化LLM客户端失败: {str(e)}")
            
        self.finished.emit()
        
    def _extract_speaker(self):
        """从样本音频中提取说话人特征"""
        sample_path = self.params.get('sample_path')
        if not sample_path or not os.path.exists(sample_path):
            self.error.emit(f"样本文件不存在: {sample_path}")
            return
            
        self.progress.emit(f"正在从音频提取说话人特征: {sample_path}")
        
        try:
            # 加载音频并提取特征
            audio_sample = load_audio(sample_path, 24000)
            
            # 只使用前30秒
            max_samples = 24000 * 30
            if len(audio_sample) > max_samples:
                self.progress.emit("音频文件较长，只使用前30秒进行声音特征提取")
                audio_sample = audio_sample[:max_samples]
                
            self.spk_smp = self.chat.sample_audio_speaker(audio_sample)
            
            # 保存特征到文件
            speaker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
            os.makedirs(speaker_dir, exist_ok=True)
            speaker_file = os.path.join(speaker_dir, "default_speaker.pt")
            torch.save(self.spk_smp, speaker_file)
            self.progress.emit(f"已保存说话人特征到: {speaker_file}")
            
            self.progress.emit("成功提取说话人特征")
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"提取说话人特征失败: {str(e)}")
            
    def _load_speaker_feature(self):
        """加载保存的说话人特征"""
        speaker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
        speaker_file = os.path.join(speaker_dir, "default_speaker.pt")
        
        if os.path.exists(speaker_file):
            try:
                self.spk_smp = torch.load(speaker_file)
                self.progress.emit(f"已加载说话人特征: {speaker_file}")
                return True
            except Exception as e:
                self.error.emit(f"加载说话人特征失败: {str(e)}")
                return False
        else:
            self.progress.emit("未找到保存的说话人特征，需要重新提取")
            return False
            
    def _generate_audio(self):
        """生成音频"""
        text = self.params.get('text', '')
        
        if not text:
            self.error.emit("请输入要转换为语音的文本")
            return
            
        # 尝试加载保存的说话人特征
        if not self.spk_smp and not self._load_speaker_feature():
            self.error.emit("请先提取说话人特征")
            return
            
        self.progress.emit("正在生成音频...")
        
        try:
            # 主动清理GPU内存
            torch.cuda.empty_cache()
            self.progress.emit("已清理GPU内存缓存")
            
            # 设置推理参数
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_smp=self.spk_smp,
                txt_smp="音频样本的文本转写内容"
            )
            
            # 检查文本长度，如果较长则分段处理
            max_chunk_length = 100  # 每段最大字符数
            if len(text) > max_chunk_length:
                self.progress.emit(f"文本较长 ({len(text)} 字符)，使用分段生成...")
                
                # 分段处理文本
                chunks = self._split_text_into_chunks(text, max_chunk_length)
                self.progress.emit(f"文本已分为 {len(chunks)} 段")
                
                # 存储各段生成的音频
                wav_chunks = []
                
                # 逐段生成音频
                for i, chunk in enumerate(chunks):
                    self.progress.emit(f"正在处理第 {i+1}/{len(chunks)} 段文本...")
                    
                    # 每次生成前清理GPU内存
                    torch.cuda.empty_cache()
                    
                    # 生成当前文本段的音频
                    chunk_wav = self.chat.infer(chunk, params_infer_code=params_infer_code)
                    
                    if len(chunk_wav) > 0:
                        wav_chunks.append(chunk_wav[0])
                    else:
                        self.error.emit(f"第 {i+1} 段文本生成的音频为空")
                
                # 合并所有音频段
                if wav_chunks:
                    combined_wav = np.concatenate(wav_chunks, axis=0)
                    self.progress.emit("音频片段合并成功")
                    self.audio_ready.emit(combined_wav)
                else:
                    self.error.emit("生成的所有音频片段均为空")
            else:
                # 短文本直接生成
                wav = self.chat.infer(text, params_infer_code=params_infer_code)
                
                if len(wav) > 0:
                    self.progress.emit("音频生成成功")
                    self.audio_ready.emit(wav[0])
                else:
                    self.error.emit("生成的音频为空")
                    
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"生成音频失败: {str(e)}")
            # 尝试清理内存
            torch.cuda.empty_cache()
            
    def _llm_chat(self):
        """与LLM对话"""
        user_input = self.params.get('user_input', '')
        
        if not user_input:
            self.error.emit("请输入问题")
            return
            
        if not self.llm_client:
            self.error.emit("LLM客户端未初始化")
            return
            
        # 尝试加载保存的说话人特征
        if not self.spk_smp and not self._load_speaker_feature():
            self.error.emit("请先提取说话人特征")
            return
            
        self.progress.emit("正在与LLM对话...")
        
        try:
            # 获取LLM回答
            response = self.llm_client.chat.completions.create(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": "你是一个有帮助的智能助手。请提供有用、安全、准确的信息，并使用中文回答问题。"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            
            # 获取回复文本
            response_text = response.choices[0].message.content
            self.llm_response.emit(response_text)
            
            # 主动清理GPU内存
            torch.cuda.empty_cache()
            self.progress.emit("已清理GPU内存缓存，准备生成语音...")
            
            # 设置推理参数
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_smp=self.spk_smp,
                txt_smp="音频样本的文本转写内容"
            )
            
            # 检查回答长度，如果较长则分段处理
            max_chunk_length = 100  # 每段最大字符数
            if len(response_text) > max_chunk_length:
                self.progress.emit(f"回答较长 ({len(response_text)} 字符)，使用分段生成...")
                
                # 分段处理文本
                chunks = self._split_text_into_chunks(response_text, max_chunk_length)
                self.progress.emit(f"回答已分为 {len(chunks)} 段")
                
                # 存储各段生成的音频
                wav_chunks = []
                
                # 逐段生成音频
                for i, chunk in enumerate(chunks):
                    self.progress.emit(f"正在处理第 {i+1}/{len(chunks)} 段文本...")
                    
                    # 每次生成前清理GPU内存
                    torch.cuda.empty_cache()
                    
                    # 生成当前文本段的音频
                    chunk_wav = self.chat.infer(chunk, params_infer_code=params_infer_code)
                    
                    if len(chunk_wav) > 0:
                        wav_chunks.append(chunk_wav[0])
                    else:
                        self.error.emit(f"第 {i+1} 段文本生成的音频为空")
                
                # 合并所有音频段
                if wav_chunks:
                    combined_wav = np.concatenate(wav_chunks, axis=0)
                    self.progress.emit("音频片段合并成功")
                    self.audio_ready.emit(combined_wav)
                else:
                    self.error.emit("生成的所有音频片段均为空")
            else:
                # 短文本直接生成
                wav = self.chat.infer(response_text, params_infer_code=params_infer_code)
                
                if len(wav) > 0:
                    self.progress.emit("语音生成成功")
                    self.audio_ready.emit(wav[0])
                else:
                    self.error.emit("生成的语音为空")
                    
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"LLM对话失败: {str(e)}")
            # 尝试清理内存
            torch.cuda.empty_cache()
    
    def _split_text_into_chunks(self, text, max_chunk_length):
        """
        将文本分割成适合处理的小段
        尝试在句子边界切分，避免在单词或中文字符中间切分
        """
        # 定义句子结束标记
        sentence_ends = ['.', '!', '?', '。', '！', '？', '\n']
        
        chunks = []
        current_chunk = ""
        
        for char in text:
            current_chunk += char
            
            # 当前块达到最大长度且在句子结束标记处
            if len(current_chunk) >= max_chunk_length and current_chunk[-1] in sentence_ends:
                chunks.append(current_chunk)
                current_chunk = ""
        
        # 添加最后一块(如果有)
        if current_chunk:
            chunks.append(current_chunk)
        
        # 如果没有找到适合的句子结束点，强制按长度分割
        if not chunks:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        return chunks

class AudioPlayer:
    """用于播放音频的类"""
    
    def __init__(self):
        """初始化音频播放器"""
        self.pyaudio_instance = None
        if AUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()
        
    def play_audio(self, audio_data, sample_rate=24000):
        """播放音频数据"""
        if not AUDIO_AVAILABLE or self.pyaudio_instance is None:
            logger.warning("PyAudio不可用，无法播放音频")
            return False
            
        try:
            # 确保是numpy数组
            if torch.is_tensor(audio_data):
                audio_np = audio_data.cpu().numpy()
            else:
                audio_np = np.array(audio_data)
                
            # 确保音频是float32类型且范围在[-1, 1]之间
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
                
            # 转换为16位整数PCM格式
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # 开始播放
            stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(2),  # 16位 = 2字节
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            # 写入数据
            stream.write(audio_int16.tobytes())
            
            # 关闭流
            stream.stop_stream()
            stream.close()
            
            return True
        except Exception as e:
            logger.error(f"播放音频失败: {str(e)}")
            return False
            
    def close(self):
        """关闭PyAudio实例"""
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

def save_as_mp3(audio_tensor, sample_rate, file_path):
    """将音频张量保存为MP3文件"""
    try:
        from pydub import AudioSegment
        import io
        
        # 将Torch张量转换为numpy数组
        if torch.is_tensor(audio_tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = np.array(audio_tensor)
        
        # 确保音频是float32类型且范围在[-1, 1]之间
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # 转换为16位整数PCM格式
        audio_np_int16 = (audio_np * 32767).astype(np.int16)
        
        # 创建AudioSegment
        audio_segment = AudioSegment(
            audio_np_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16位 = 2字节
            channels=1 if len(audio_np_int16.shape) == 1 else audio_np_int16.shape[1]
        )
        
        # 导出为MP3
        audio_segment.export(file_path, format="mp3")
        return True
    except Exception as e:
        logger.error(f"保存MP3失败: {str(e)}")
        return False

def save_as_wav(audio_data, sample_rate, file_path):
    """将音频数据保存为WAV文件(处理torchaudio可能的问题)"""
    try:
        # 确保是numpy数组
        if torch.is_tensor(audio_data):
            audio_np = audio_data.cpu().numpy()
        else:
            audio_np = np.array(audio_data)
            
        # 确保音频是float32类型且范围在[-1, 1]之间
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
            
        # 转换为16位整数PCM格式
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # 创建wave文件
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16位 = 2字节
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return True
    except Exception as e:
        logger.error(f"保存WAV失败: {str(e)}")
        # 尝试使用torchaudio保存
        try:
            # 转换为torch张量
            if not torch.is_tensor(audio_data):
                audio_tensor = torch.from_numpy(np.array(audio_data, dtype=np.float32))
            else:
                audio_tensor = audio_data
                
            # 确保形状正确 [channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
                
            # 保存为WAV
            torchaudio.save(file_path, audio_tensor, sample_rate)
            return True
        except Exception as sub_e:
            logger.error(f"使用torchaudio保存WAV也失败: {str(sub_e)}")
            return False

class ChatTTSGUI(QMainWindow):
    """ChatTTS GUI主窗口"""
    
    def __init__(self):
        """初始化主窗口"""
        # 确保环境变量设置在任何PyTorch操作之前
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("ChatTTS GUI")
        self.setMinimumSize(800, 600)
        
        # 初始化成员变量
        self.chat = None
        self.llm_client = None
        self.worker_thread = None
        self.audio_player = AudioPlayer()
        self.current_audio = None
        self.spk_smp = None
        self.sample_path = ""
        
        # 设置UI
        self.init_ui()
        
        # 加载模型
        self.load_model()
        
    def init_ui(self):
        """初始化UI"""
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建状态面板
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        
        self.status_label = QLabel("状态: 未初始化")
        status_layout.addWidget(self.status_label)
        
        self.sample_label = QLabel("样本: 未选择")
        status_layout.addWidget(self.sample_label)
        
        main_layout.addWidget(status_frame)
        
        # 创建模式选择标签页
        tab_widget = QTabWidget()
        
        # 添加Generate Audio模式标签页
        generate_tab = self.create_generate_tab()
        tab_widget.addTab(generate_tab, "Generate Audio Mode")
        
        # 添加Dialogue模式标签页
        dialogue_tab = self.create_dialogue_tab()
        tab_widget.addTab(dialogue_tab, "Dialogue Mode")
        
        main_layout.addWidget(tab_widget)
        
    def create_generate_tab(self):
        """创建Generate Audio模式的标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 样本选择部分
        sample_frame = QFrame()
        sample_frame.setFrameShape(QFrame.StyledPanel)
        sample_layout = QHBoxLayout(sample_frame)
        
        sample_layout.addWidget(QLabel("样本音频:"))
        
        self.sample_path_edit = QLineEdit()
        self.sample_path_edit.setReadOnly(True)
        sample_layout.addWidget(self.sample_path_edit)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_sample)
        sample_layout.addWidget(browse_button)
        
        extract_button = QPushButton("提取特征")
        extract_button.clicked.connect(self.extract_speaker)
        sample_layout.addWidget(extract_button)
        
        layout.addWidget(sample_frame)
        
        # 文本输入部分
        text_frame = QFrame()
        text_frame.setFrameShape(QFrame.StyledPanel)
        text_layout = QVBoxLayout(text_frame)
        
        text_layout.addWidget(QLabel("输入要转换为语音的文本:"))
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("在此输入文本...")
        text_layout.addWidget(self.text_edit)
        
        layout.addWidget(text_frame)
        
        # 生成和播放按钮
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        generate_button = QPushButton("生成语音")
        generate_button.clicked.connect(self.generate_audio)
        button_layout.addWidget(generate_button)
        
        play_button = QPushButton("播放语音")
        play_button.clicked.connect(self.play_audio)
        button_layout.addWidget(play_button)
        
        save_wav_button = QPushButton("保存为WAV")
        save_wav_button.clicked.connect(self.save_audio_wav)
        button_layout.addWidget(save_wav_button)
        
        save_mp3_button = QPushButton("保存为MP3")
        save_mp3_button.clicked.connect(self.save_audio_mp3)
        button_layout.addWidget(save_mp3_button)
        
        layout.addWidget(button_frame)
        
        # 日志显示
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        
        log_layout.addWidget(QLabel("日志:"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_frame)
        
        return tab
        
    def create_dialogue_tab(self):
        """创建Dialogue模式的标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 对话历史
        chat_frame = QFrame()
        chat_frame.setFrameShape(QFrame.StyledPanel)
        chat_layout = QVBoxLayout(chat_frame)
        
        chat_layout.addWidget(QLabel("对话历史:"))
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        chat_layout.addWidget(self.chat_history)
        
        layout.addWidget(chat_frame)
        
        # 用户输入
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_layout = QHBoxLayout(input_frame)
        
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入问题...")
        self.input_edit.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_edit)
        
        send_button = QPushButton("发送")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        layout.addWidget(input_frame)
        
        # 控制按钮
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        play_response_button = QPushButton("重播最后回答")
        play_response_button.clicked.connect(self.play_audio)
        control_layout.addWidget(play_response_button)
        
        save_wav_button = QPushButton("保存回答为WAV")
        save_wav_button.clicked.connect(self.save_audio_wav)
        control_layout.addWidget(save_wav_button)
        
        save_mp3_button = QPushButton("保存回答为MP3")
        save_mp3_button.clicked.connect(self.save_audio_mp3)
        control_layout.addWidget(save_mp3_button)
        
        clear_button = QPushButton("清除对话")
        clear_button.clicked.connect(self.clear_chat)
        control_layout.addWidget(clear_button)
        
        layout.addWidget(control_frame)
        
        return tab
        
    def log_message(self, message):
        """向日志文本框添加消息"""
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.log_text.moveCursor(QTextCursor.End)
        
    def update_status(self, message):
        """更新状态标签"""
        self.status_label.setText(f"状态: {message}")
        
    def browse_sample(self):
        """浏览选择样本音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择样本音频文件", "", "音频文件 (*.mp3 *.wav *.flac *.ogg)"
        )
        
        if file_path:
            self.sample_path = file_path
            self.sample_path_edit.setText(file_path)
            self.sample_label.setText(f"样本: {os.path.basename(file_path)}")
            self.log_message(f"已选择样本文件: {file_path}")
            
    def load_model(self):
        """加载模型"""
        self.update_status("正在加载模型...")
        
        # 确保环境变量设置在任何PyTorch操作之前
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread('load_model')
        self.worker_thread.progress.connect(self.log_message)
        self.worker_thread.error.connect(self.handle_error)
        self.worker_thread.finished.connect(self.handle_model_loaded)
        self.worker_thread.start()
        
    def extract_speaker(self):
        """提取说话人特征"""
        if not self.sample_path:
            self.handle_error("请先选择样本音频文件")
            return
            
        self.update_status("正在提取说话人特征...")
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread('extract_speaker', {'sample_path': self.sample_path})
        self.worker_thread.chat = self.chat  # 使用已加载的模型
        self.worker_thread.progress.connect(self.log_message)
        self.worker_thread.error.connect(self.handle_error)
        self.worker_thread.finished.connect(self.handle_speaker_extracted)
        self.worker_thread.start()
        
    def generate_audio(self):
        """生成音频"""
        text = self.text_edit.toPlainText().strip()
        
        if not text:
            self.handle_error("请输入要转换为语音的文本")
            return
            
        self.update_status("正在生成音频...")
        
        # 创建并启动工作线程
        worker = WorkerThread('generate_audio', {'text': text})
        worker.chat = self.chat  # 使用已加载的模型
        if hasattr(self.worker_thread, 'spk_smp') and self.worker_thread.spk_smp is not None:
            worker.spk_smp = self.worker_thread.spk_smp  # 使用已提取的特征
        worker.progress.connect(self.log_message)
        worker.error.connect(self.handle_error)
        worker.audio_ready.connect(self.handle_audio_ready)
        worker.finished.connect(lambda: self.update_status("音频生成完成"))
        self.worker_thread = worker  # 更新当前工作线程
        worker.start()
        
    def send_message(self):
        """发送消息给LLM"""
        user_input = self.input_edit.text().strip()
        
        if not user_input:
            return
            
        if not hasattr(self.worker_thread, 'llm_client') or self.worker_thread.llm_client is None:
            self.handle_error("LLM客户端未初始化")
            return
            
        # 添加用户消息到对话历史
        self.chat_history.append(f"<b>You:</b> {user_input}")
        self.input_edit.clear()
        
        self.update_status("正在与LLM对话...")
        
        # 创建并启动工作线程
        worker = WorkerThread('llm_chat', {'user_input': user_input})
        worker.chat = self.chat  # 使用已加载的模型
        worker.llm_client = self.worker_thread.llm_client  # 使用已初始化的LLM客户端
        if hasattr(self.worker_thread, 'spk_smp') and self.worker_thread.spk_smp is not None:
            worker.spk_smp = self.worker_thread.spk_smp  # 使用已提取的特征
        worker.progress.connect(self.log_message)
        worker.error.connect(self.handle_error)
        worker.llm_response.connect(self.handle_llm_response)
        worker.audio_ready.connect(self.handle_audio_ready)
        worker.finished.connect(lambda: self.update_status("对话完成"))
        self.worker_thread = worker  # 更新当前工作线程
        worker.start()
        
    def play_audio(self):
        """播放当前音频"""
        if self.current_audio is None:
            self.handle_error("没有可用的音频")
            return
            
        self.log_message("正在播放音频...")
        if self.audio_player.play_audio(self.current_audio):
            self.log_message("音频播放完成")
        else:
            self.handle_error("播放音频失败")
            
    def save_audio_mp3(self):
        """保存当前音频为MP3"""
        if self.current_audio is None:
            self.handle_error("没有可用的音频")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3", "MP3文件 (*.mp3)"
        )
        
        if file_path:
            self.log_message(f"正在保存音频到 {file_path}...")
            if save_as_mp3(self.current_audio, 24000, file_path):
                self.log_message(f"音频已保存到 {file_path}")
            else:
                self.handle_error("保存音频失败")
                
    def save_audio_wav(self):
        """保存当前音频为WAV"""
        if self.current_audio is None:
            self.handle_error("没有可用的音频")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav", "WAV文件 (*.wav)"
        )
        
        if file_path:
            self.log_message(f"正在保存音频到 {file_path}...")
            if save_as_wav(self.current_audio, 24000, file_path):
                self.log_message(f"音频已保存到 {file_path}")
            else:
                self.handle_error("保存音频失败")
                
    def clear_chat(self):
        """清除对话历史"""
        self.chat_history.clear()
        self.log_message("对话历史已清除")
        
    def handle_error(self, error_message):
        """处理错误消息"""
        self.log_message(f"错误: {error_message}")
        self.update_status("错误")
        
    def handle_model_loaded(self):
        """处理模型加载完成事件"""
        self.chat = self.worker_thread.chat
        self.llm_client = self.worker_thread.llm_client
        self.update_status("就绪")
        self.log_message("模型加载完成，系统就绪")
        
    def handle_speaker_extracted(self):
        """处理说话人特征提取完成事件"""
        self.spk_smp = self.worker_thread.spk_smp
        self.update_status("特征提取完成")
        self.log_message("说话人特征提取完成")
        
    def handle_audio_ready(self, audio_data):
        """处理音频生成完成事件"""
        self.current_audio = audio_data
        self.play_audio()
        
    def handle_llm_response(self, response):
        """处理LLM回答"""
        self.chat_history.append(f"<b>AI:</b> {response}")
        self.chat_history.moveCursor(QTextCursor.End)
        
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 清理资源
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
            
        self.audio_player.close()
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，在所有平台上看起来更一致
    
    # 设置字体
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = ChatTTSGUI()
    window.show()
    
    # 开始应用程序事件循环
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()