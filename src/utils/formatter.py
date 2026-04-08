from PIL import Image
import json
from typing import Dict, Any

def make_conversation(example):
    # 适配当前数据集结构
    # 数据集已经有 messages 字段包含对话格式
    try:
        # 直接使用数据集中的 messages 字段
        messages = example.get("messages", [])

        # 处理图片
        images = example.get("images", [])
        return {
            # 关键点：prompt 是"对话列表"，不是字符串
            "prompt": messages,
            "image": images,
        }
    except Exception as e:
        # 出错时返回基本信息
        return {
            "prompt": example.get("messages", []),
            "image": example.get("images", []),
        }

def format_url(image_path: str, local=False) -> str:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    else:
        # 本地文件，转换为 data URI
        with open(image_path, "rb") as f:
            b = f.read()
        import base64
        b64 = base64.b64encode(b).decode("ascii")
        # 根据图片类型调整 mime（此处假设 png）
        data_uri = f"data:image/png;base64,{b64}"
        return data_uri

def add_row(sample,name,value):
    sample[name] = value
    return sample