from openai import OpenAI

prompt_dict = {
    "kimi": [
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。",
        },
        {
            "role": "user",
            "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。",
        },
        {
            "role": "assistant",
            "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。",
        },
    ],
    "deepseek": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。",
        },
        {
            "role": "assistant",
            "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。",
        },
    ],
    "deepseek_TN": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "你好，现在我们在处理TTS的文本输入，下面将会给你输入一段文本，请你将其中的阿拉伯数字等等转为文字表达，并且输出的文本里仅包含逗号和句号这两个标点符号",
        },
        {
            "role": "assistant",
            "content": "好的，我现在对TTS的文本输入进行处理。这一般叫做text normalization。下面请输入",
        },
        {"role": "user", "content": "We paid $123 for this desk."},
        {
            "role": "assistant",
            "content": "We paid one hundred and twenty three dollars for this desk.",
        },
        {"role": "user", "content": "详询请拨打010-724654"},
        {"role": "assistant", "content": "详询请拨打零幺零，七二四六五四"},
        {"role": "user", "content": "罗森宣布将于7月24日退市，在华门店超6000家！"},
        {
            "role": "assistant",
            "content": "罗森宣布将于七月二十四日退市，在华门店超过六千家。",
        },
    ],
}


import aiohttp
import backoff
import asyncio
import json
from typing import Dict, Optional
import logging

class ChatOpenAI:
    """处理本地LLM对话的客户端类"""
    
    def __init__(self, base_url: str, model: str):
        """
        初始化ChatOpenAI客户端
        
        参数:
            base_url: API基础URL
            model: 要使用的模型名称
        """
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger("ChatTTS_GUI.LLM")
        
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def async_call(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        异步调用LLM获取回答
        
        参数:
            user_input: 用户输入文本
            system_prompt: 系统提示（可选）
            
        返回:
            LLM的回答文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['message']['content']
                    else:
                        error_text = await response.text()
                        raise Exception(f"API Error: {response.status} - {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Error in LLM call: {str(e)}")
            raise
            
    def call(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        同步调用LLM获取回答（在内部使用异步调用）
        
        参数:
            user_input: 用户输入文本
            system_prompt: 系统提示（可选）
            
        返回:
            LLM的回答文本
        """
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 在事件循环中运行异步调用
            response = loop.run_until_complete(self.async_call(user_input, system_prompt))
            return response
        finally:
            # 关闭事件循环
            loop.close()
