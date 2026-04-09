import re
import ast
import json
from string import Template
from typing import List, Callable, Tuple, Any, Dict
from dataclasses import dataclass
from types import MethodType

from zai import ZhipuAiClient
import torch

import inspect
import platform
import numpy as np

import os
from src.config import SETTINGS, require_zhipuai_api_key

from transformers import AutoProcessor, AutoModelForVision2Seq, Glm4vForConditionalGeneration, AutoModelForImageTextToText


class BASE:
    def __init__(self, 
                 model_name: str,
                 SYSTEM_PROMPT: str=None,
                 tools: List[Callable]=[],
                 device: str=None
                 ):
        self.tools = { func.__name__: func for func in (tools or []) }
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Token usage tracking
        self.total_tokens = 0
        self.last_call_tokens = 0


    def update_system_prompt(self, new_system_prompt: str):
        self.system_prompt = new_system_prompt

    def run(self, user_input: str|None=None, image_paths: List[str]|None=None):
        pass

    def call_model(self, messages):
        pass

        
class GPT(BASE):
    def __init__(self, 
                 tools: List[Callable], 
                 model_name: str,
                 ):
        super().__init__(tools, model_name)

class GEMINI(BASE):
    def __init__(self, 
                 tools: List[Callable], 
                 model_name: str,
                 ):
        super().__init__(tools, model_name)

    def call_model(self, messages):
        response = self.client.chat.completions.create(
                model_name=self.model_name,
                messages=messages,
            )
        
class Local(BASE):
    def __init__(self, model_name: str = "Qwen2.5-VL-3B-Instruct", SYSTEM_PROMPT: str = None, tools: List = None, model_path: str = None):
        """
        鍒濆鍖栨湰鍦癆gent
        
        Args:
            model_name: 鍩虹妯″瀷鍚嶇О
            SYSTEM_PROMPT: 绯荤粺鎻愮ず
            tools: 宸ュ叿鍒楄〃
            model_path: 璁粌濂界殑妯″瀷璺緞锛堝彲閫夛級
        """
        super().__init__(model_name, SYSTEM_PROMPT, tools or [])
        
        # 鍒濆鍖栨ā鍨嬪拰澶勭悊锟?
        self.model = None
        self.processor = None
        self.model_path = model_path
        
        # 鍔犺浇鍩虹妯″瀷
        self._load_base_model(model_name)
        
        # 濡傛灉鎻愪緵浜嗚缁冨ソ鐨勬ā鍨嬭矾寰勶紝鍔犺浇锟?
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)
        
        # 鍒濆鍖栦笂涓嬫枃
        self.context = [{"role": "system", "content": self.system_prompt or ""}]
    
    def _load_base_model(self, model_name: str):
        """Load the base model backend."""
        try:
            name_lower = model_name.lower()
            if "qwen3" in name_lower:
                print(f"Loading Qwen3 model: {model_name}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    padding_side="left",
                )
            elif "qwen2.5" in name_lower or "qwen2_5" in name_lower or "qwen2.5-vl" in name_lower:
                # 鍚戝悗鍏煎鏃х殑 Qwen2.5 娴佺▼
                from transformers import Qwen2_5_VLForConditionalGeneration
                print(f"Loading Qwen2.5 model: {model_name}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    padding_side="left",
                )
            elif "ui-tars" in name_lower or "ui_tars" in name_lower:
                # 鏀寔 UI-TARS 绯诲垪妯″瀷浣滀负鏈湴鍩哄骇
                print(f"Loading UI-TARS model: {model_name}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    padding_side="left",
                    trust_remote_code=True,
                )
            else:
                print(f"鈿狅笍  Unsupported model: {model_name}")
                return

            # 涓烘湰鍦板妯℃€佹ā鍨嬬粺涓€娣诲姞瀹夊叏鐨勬壒澶勭悊瑙ｇ爜
            def safe_batch_decode(self, sequences, **kwargs):
                pad_id = self.tokenizer.pad_token_id or 0
                vocab_size = len(self.tokenizer)

                def clean_one(seq):
                    if isinstance(seq, torch.Tensor):
                        ids = seq.clone().detach().cpu().tolist()
                    else:
                        ids = list(seq)

                    cleaned = []
                    for x in ids:
                        try:
                            v = int(x)
                        except Exception:
                            v = pad_id
                        if v < 0 or v >= vocab_size:
                            v = pad_id
                        cleaned.append(v)
                    return cleaned

                if isinstance(sequences, torch.Tensor):
                    cleaned_batch = [clean_one(row) for row in sequences]
                else:
                    cleaned_batch = [clean_one(row) for row in sequences]

                return self.tokenizer.batch_decode(cleaned_batch, **kwargs)

            if self.processor is not None:
                self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

            print("锟?Base model loaded successfully")
        except Exception as e:
            print(f"锟?Error loading base model: {e}")
            raise
    
    def _load_finetuned_model(self, model_path: str):
        """Load finetuned weights on top of the current base model."""
        try:
            from peft import PeftModel
            print(f"Loading finetuned model from: {model_path}")

            base_name = (
                self.model.config._name_or_path
                if hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path")
                else self.model_name
            )
            lower_name = str(base_name).lower()

            if "ui-tars" in lower_name or "ui_tars" in lower_name:
                # UI-TARS 绯诲垪锛氫娇锟?AutoModelForVision2Seq 浣滀负鍩哄骇
                print(f"Loading UI-TARS base model for finetuned weights: {base_name}")
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            elif "qwen3" in lower_name:
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
            else:
                # 榛樿鍥為€€锟?Qwen2.5
                from transformers import Qwen2_5_VLForConditionalGeneration
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base_name if base_name else "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )

            # 鍔犺浇 Peft 妯″瀷
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("锟?Finetuned model loaded successfully")

        except Exception as e:
            print(f"锟?Error loading finetuned model: {e}")
            print("鈿狅笍  Using base model instead")
    
    def call_model(self, messages):
        """璋冪敤妯″瀷鐢熸垚鍝嶅簲"""
        try:
            if self.model is None or self.processor is None:
                return "Error: Model not loaded"
            
            # 搴旂敤鑱婂ぉ妯℃澘
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # 閫掑綊鍦板皢鎵€锟?tensor 绉诲埌妯″瀷鎵€鍦ㄨ澶囷紝閬垮厤璁惧涓嶄竴鑷村鑷寸殑閿欒
            def move_to_device(obj, device):
                import torch
                import numpy as _np

                # Tensor -> move
                if torch.is_tensor(obj):
                    return obj.to(device)

                # numpy array -> convert then move
                if isinstance(obj, _np.ndarray):
                    try:
                        return torch.from_numpy(obj).to(device)
                    except Exception:
                        return obj

                # list of scalars -> convert to tensor
                if isinstance(obj, list) and len(obj) > 0 and all(not isinstance(x, (dict, list, tuple)) for x in obj):
                    try:
                        return torch.tensor(obj).to(device)
                    except Exception:
                        pass

                # dict/list/tuple -> recurse
                if isinstance(obj, dict):
                    return {k: move_to_device(v, device) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    cls = list if isinstance(obj, list) else tuple
                    return cls(move_to_device(v, device) for v in obj)

                return obj

            try:
                model_device = None
                try:
                    model_device = next(self.model.parameters()).device
                except Exception:
                    model_device = getattr(self.model, 'device', None)

                if model_device is not None:
                    # 浼樺厛浣跨敤 BatchEncoding/瀵硅薄锟?.to(device) 鏂规硶锛堝鏋滄湁锟?
                    try:
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(model_device)
                        else:
                            inputs = move_to_device(inputs, model_device)
                    except Exception:
                        # 鍥為€€鍒伴€掑綊绉诲姩
                        inputs = move_to_device(inputs, model_device)

                    # 濡傛灉鏈変换锟?tensor 浠嶄笉锟?model_device锛屽垯鎵撳嵃璁惧鍒嗗竷锛堜究浜庤皟璇曪級
                    try:
                        def collect_devices(x, prefix=""):
                            out = []
                            import torch
                            if torch.is_tensor(x):
                                out.append((prefix or "tensor", str(x.device)))
                                return out
                            if isinstance(x, dict):
                                for k, v in x.items():
                                    out.extend(collect_devices(v, prefix + "." + str(k)))
                                return out
                            if isinstance(x, (list, tuple)):
                                for i, v in enumerate(x):
                                    out.extend(collect_devices(v, prefix + f"[{i}]"))
                                return out
                            return out

                        devs = collect_devices(inputs)
                        bad = [d for d in devs if model_device is not None and d[1] != str(model_device)]
                        if bad:
                            print("鈿狅笍  Device mismatch detected in inputs (field -> device):")
                            for field, dev in devs:
                                print(f"  {field} -> {dev}")
                    except Exception:
                        pass
            except Exception:
                # 濡傛灉杩佺Щ澶辫触锛岄檷绾т负鍘熸潵鐨勬祬绉诲姩绛栫暐锛堝吋瀹规棫琛屼负锟?
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 鐢熸垚鍝嶅簲
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # 鎴柇杈撳叆閮ㄥ垎
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
            else:
                generated_ids_trimmed = generated_ids
            
            # 瑙ｇ爜鍝嶅簲
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Token counting
            input_tokens = inputs['input_ids'].numel() if 'input_ids' in inputs else 0
            # generated_ids contains both prompt and output in some flows, 
            # but generated_ids_trimmed is just the output part.
            output_tokens = sum(len(ids) for ids in generated_ids_trimmed)
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"锟?Error in model inference: {e}")
            return f"Error: {str(e)}"
    
    def run(self, user_input: str = None, image_paths: List[str] = None):
        """杩愯鎺ㄧ悊"""
        messages = []
        
        # 娣诲姞绯荤粺鎻愮ず
        system_content = [{"type": "text", "text": self.system_prompt or ""}]
        messages.append({"role": "system", "content": system_content})
        
        # 娣诲姞鐢ㄦ埛杈撳叆
        user_content = []
        
        # 澶勭悊鍥剧墖
        if image_paths:
            for img_path in image_paths:
                # 妫€鏌ュ浘鐗囪矾寰勬槸鍚﹀瓨锟?
                if os.path.exists(img_path):
                    user_content.append({
                        "type": "image",
                        "image": img_path
                    })
                else:
                    print(f"鈿狅笍 Image not found: {img_path[:10]}")
        
        # 澶勭悊鏂囨湰杈撳叆
        if user_input:
            user_content.append({
                "type": "text",
                "text": user_input
            })
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        # 璋冪敤妯″瀷
        response = self.call_model(messages)
        
        # 鏇存柊涓婁笅锟?
        self.context.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": response}]
        })
        
        return response
    
    def update_system_prompt(self, new_system_prompt: str):
        """鏇存柊绯荤粺鎻愮ず"""
        self.system_prompt = new_system_prompt
        
        # 鏇存柊涓婁笅鏂囩殑绗竴涓秷锟?
        if self.context and self.context[0].get("role") == "system":
            self.context[0] = {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]}
        else:
            self.context.insert(0, {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]})
        
        print(f"锟?System prompt updated")
    
    def load_model_from_path(self, model_path: str):
        """Load a finetuned model from a local path."""
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)
            self.model_path = model_path
        else:
            print(f"锟?Model path not found or invalid: {model_path}")
    
    def get_model_info(self) -> dict:
        """鑾峰彇妯″瀷淇℃伅"""
        info = {
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "model_path": self.model_path,
            "device": str(self.model.device) if self.model and hasattr(self.model, 'device') else "unknown"
        }
        
        if self.model and hasattr(self.model, 'config'):
            info["model_name"] = self.model.config._name_or_path
        
        return info
        

class GLM(BASE):
    def __init__(self, model_name:str, api_key:str=None, SYSTEM_PROMPT:str=None,tools:List=None,):
        super().__init__(model_name, SYSTEM_PROMPT, tools)
        self.api_key = require_zhipuai_api_key(api_key)`r`n        self.client = ZhipuAiClient(api_key=self.api_key)  # 濉啓鎮ㄨ嚜宸辩殑 APIKey



    def call_model(self, messages,think=False):
        if think:
            thinking_mode = "enabled"
        else:
            thinking_mode = "disabled"
        
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                thinking={
                    "type":thinking_mode
                }
            )

        # Token counting
        if hasattr(response, 'usage') and response.usage:
            self.last_call_tokens = getattr(response.usage, 'total_tokens', 0)
            self.total_tokens += self.last_call_tokens

        return response.choices[0].message.content


class Qwen3VLBackend(BASE):
    
    """Backend wrapper for local Qwen3-VL models used by ReActAgent.

    Provides a unified call_model(messages) -> str interface.
    """

    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 鑷姩閫夋嫨璁惧锛氫紭鍏堜娇鐢ㄤ紶鍏ョ殑 device锛屽叾娆℃牴鎹幆澧冩锟?
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading Qwen3 VL model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("锟?Qwen3 VL model loaded successfully")
        except Exception as e:
            print(f"锟?Error loading Qwen3 VL model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the Qwen3-VL input format."""

        涓昏澶勭悊锟?
        - system.content 鑻ヤ负瀛楃涓诧紝鏀逛负 [{"type":"text","text":...}]
        - user.content 涓殑 {"type":"image_url","image_url":{"url":...}} -> {"type":"image","url":...}
        """
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 缁熶竴锟?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    # 鏉ヨ嚜 ReActAgent.run: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    # 鍏煎宸叉湁锟?{"type":"image","image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    else:
                        # 鍏跺畠淇濇寔鍘熸牱锛堝 {"type":"text",...}锟?
                        new_content.append(item)
            else:
                # 涓嶆敮鎸佺殑 content 缁撴瀯锛岃烦锟?
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: Qwen3 VL model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 灏嗚緭鍏ョЩ鍒版ā鍨嬫墍鍦ㄨ锟?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 鍙В鐮佹柊鐢熸垚鐨勯儴锟?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"锟?Error in Qwen3 VL inference: {e}")
            return f"Error: {str(e)}"


class UITARSBackend(BASE):
    """Backend wrapper for ByteDance UI-TARS-1.5-7B."""

    def __init__(
        self,
        model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 鑷姩閫夋嫨璁惧
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading UI-TARS model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            print("锟?UI-TARS model loaded successfully")
        except Exception as e:
            print(f"锟?Error loading UI-TARS model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the UI-TARS input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 缁熶竴锟?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # ReActAgent 鏍煎紡: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # 鍏煎鏍煎紡: {"type": "image", "image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # 鍏煎鏍煎紡: {"type": "image", "url": path}
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # 鏂囨湰鎴栧叾浠栦繚鎸佸師锟?
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: UI-TARS model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 灏嗚緭鍏ョЩ鍒版ā鍨嬫墍鍦ㄨ锟?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 鍙В鐮佹柊鐢熸垚鐨勯儴锟?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"锟?Error in UI-TARS inference: {e}")
            return f"Error: {str(e)}"


class GLMFlashBackend(BASE):
    """Backend wrapper for zai-org/GLM-4.6V-Flash."""

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.6V-Flash",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 鑷姩閫夋嫨璁惧
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading GLM-Flash model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            self.model.eval()
            print("锟?GLM-Flash model loaded successfully")
        except Exception as e:
            print(f"锟?Error loading GLM-Flash model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the GLM-Flash input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 缁熶竴锟?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # ReActAgent 鏍煎紡: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # 鍏煎鏍煎紡: {"type": "image", "image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # 鍏煎鏍煎紡: {"type": "image", "url": path}
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # 鏂囨湰鎴栧叾浠栦繚鎸佸師锟?
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: GLM-Flash model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # TODO: GLM-4V models usually don't need token_type_ids
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 鍙В鐮佹柊鐢熸垚鐨勯儴锟?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"锟?Error in GLM-Flash inference: {e}")
            return f"Error: {str(e)}"
    
    
class Holo2Backend(BASE):
    """Backend wrapper for Hcompany/Holo2-4B."""

    def __init__(
        self,
        model_name: str = "Hcompany/Holo2-4B",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 鑷姩閫夋嫨璁惧
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading Holo2 model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("锟?Holo2 model loaded successfully")
        except Exception as e:
            print(f"锟?Error loading Holo2 model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the Holo2 input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    else:
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: Holo2 model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 灏嗚緭鍏ョЩ鍒版ā鍨嬫墍鍦ㄨ锟?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 鍙В鐮佹柊鐢熸垚鐨勯儴锟?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            
            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens
            
            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # --- Holo2 Format Translation (JSON to XML) ---
            # Holo2 often outputs JSON like {"click": [x, y]} instead of <action> tags.
            # We translate it here to keep ReActAgent clean.
            import re
            # Extract numbers from patterns like "click": [818, 895] or "click": ["818", "895"]
            json_click_pattern = r'["\']click["\']\s*:\s*\[\s*["\']?(?P<x>\d+)["\']?,\s*["\']?(?P<y>\d+)["\']?\s*\]'
            match = re.search(json_click_pattern, text)
            if match and "<action>" not in text:
                x, y = match.group("x"), match.group("y")
                text = f"<thought>Detected Holo2 JSON output, translating to XML.</thought>\n<action>click({x}, {y})</action>"
            # -----------------------------------------------

            return text
        except Exception as e:
            print(f"锟?Error in Holo2 inference: {e}")
            return f"Error: {str(e)}"

