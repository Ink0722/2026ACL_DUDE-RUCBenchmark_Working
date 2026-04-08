from .model import Local,GLM
from .parser import parse_agent_output
from .prompt_template import static_template
from numpy import random

import base64
import os

import ast
import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple, Any, Dict
import json

ZHIPUAI_API_KEY = "dfb029674361439faae28caa78e61ecb.JLb0zzJ5KCOKwwKn"

class ReActAgent:
    def __init__(
        self,
        tools: List[Callable],
        model: str,
        project_directory: str,
        backend: str = "uitars",
        device: str | None = None,
    ):
        """通用 ReAct Agent。

        参数：
        - tools: 提供给 LLM 调用的工具函数列表
        - model: 模型名称（例如 "glm-4.6v" 或 "Qwen/Qwen3-VL-4B-Instruct"）
        - project_directory: 当前项目目录
        - backend: "glm" 使用远程 GLM API；"qwen3_local" 使用本地 Qwen3-VL 后端
        - device: 本地模型使用的设备（如 "cuda" 或 "cpu"），为空则由后端自行选择
        """
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.backend = backend
        self.device = device

        if backend == "glm":
            self.client = GLM(
                model_name=self.model,
                api_key=ZHIPUAI_API_KEY,
                tools=[],
            )
        elif backend == "qwen3_local":
            from .model import Qwen3VLBackend

            self.client = Qwen3VLBackend(
                model_name=self.model,
                SYSTEM_PROMPT=None,
                tools=[],
                device=self.device,
            )
        elif backend == "uitars":
            from .model import UITARSBackend

            self.client = UITARSBackend(
                model_name=self.model,
                SYSTEM_PROMPT=None,
                tools=[],
                device=self.device,
            )
        elif backend == "glm_flash":
            from .model import GLMFlashBackend

            self.client = GLMFlashBackend(
                model_name=self.model,
                SYSTEM_PROMPT=None,
                tools=[],
                device=self.device,
            )
        elif backend == "holo2":
            from .model import Holo2Backend

            self.client = Holo2Backend(
                model_name=self.model,
                SYSTEM_PROMPT=None,
                tools=[],
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.experience = "To be update"

    def run(self, user_input: str|None=None, image_paths: List[str]|None=None, max_steps: int=3):
        messages = []
        step_count = 0
        # 添加系统提示
        messages.append({
            "role": "system",
            "content": self.render_system_prompt(static_template)
        })
        
        # 添加用户输入
        user_content = []
        if image_paths is not None:
            for img_path in image_paths:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_path
                    }
                })
        if user_input is not None:
            user_content.append({
                "type": "text",
                "text": user_input
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })

        while True:

            # 步数限制：每次向模型发起一次调用，都算作一步
            if step_count >= max_steps:
                print(f"\n[Stop] Max steps {max_steps} reached.")
                return "Max steps reached"

            # 请求模型
            content = self.client.call_model(messages)
            step_count += 1

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                try:
                    final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                    return final_answer.group(1)
                except AttributeError: # solely <> without </>
                    final_answer = re.search(r"<final_answer>(.*?)", content, re.DOTALL)
                    print("[Error]: solely <final_answer> without </>")
                    return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                print("[Error]: Unmatch")
                print("[UNmatch]:",content)
                continue
                # raise RuntimeError("模型未输出 <action>")

            action = action_match.group(1)
            tool_name, args, kwargs = self.parse_action(action)
            print(f"🔧 Parameter Phase Action: {tool_name}")

            # 打印参数（避免 list/dict 之类 join 出错）
            pos_str = ", ".join(repr(a) for a in args)
            kw_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_str = ", ".join(s for s in [pos_str, kw_str] if s)  # 兼容没有 kwargs 的情况
            print(f"🔧 Action: {tool_name}({all_str})")
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                observation = self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"🔍 Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})


    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            # operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            experience=self.experience
            # file_list=file_list
        )

    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            return  "Final Answer should be provided instead of action" , [], {}

        func_name = match.group(1)
        args_str = match.group(2).strip()

        args = []
        kwargs = {}

        current = ""
        in_string = False
        string_char = None
        paren = bracket = brace = 0
        i = 0

        def flush_token(token):
            token = token.strip()
            if not token:
                return

            # keyword argument?  key=value
            if '=' in token and not token.startswith(('{"', "{'", "[")):  
                # split only at top-level '='
                key, value = token.split('=', 1)
                key = key.strip()
                kwargs[key] = self._parse_single_arg(value.strip())
            else:
                args.append(self._parse_single_arg(token))

        while i < len(args_str):
            ch = args_str[i]

            if not in_string:
                if ch in ['"', "'"]:
                    in_string = True
                    string_char = ch
                    current += ch
                elif ch == '(':
                    paren += 1; current += ch
                elif ch == ')':
                    paren -= 1; current += ch
                elif ch == '[':
                    bracket += 1; current += ch
                elif ch == ']':
                    bracket -= 1; current += ch
                elif ch == '{':
                    brace += 1; current += ch
                elif ch == '}':
                    brace -= 1; current += ch
                elif ch == ',' and paren == bracket == brace == 0:
                    flush_token(current)
                    current = ""
                else:
                    current += ch
            else:
                current += ch
                if ch == string_char and args_str[i-1] != '\\':
                    in_string = False
                    string_char = None

            i += 1

        if current.strip():
            flush_token(current)

        return func_name, args, kwargs

    def _parse_single_arg(self, arg_str: str):
        arg_str = arg_str.strip()

        # 判断是否为字符串字面量，包括被 LLM 转义成 \"...\" 的情况
        # 情况1: "China"
        # 情况2: \"China\"
        if (
            (arg_str.startswith('"') and arg_str.endswith('"')) or
            (arg_str.startswith('\\"') and arg_str.endswith('\\"'))
        ):
            # 去除最外层引号（处理多次转义）
            s = arg_str

            # Case like \"China\" -> strip first and last \"
            if s.startswith('\\"') and s.endswith('\\"'):
                s = s[2:-2]

            # Case like "China" -> strip quotes normally
            elif s.startswith('"') and s.endswith('"'):
                s = s[1:-1]

            # 把内部的转义字符标准化
            s = s.replace('\\"', '"')
            s = s.replace("\\'", "'")
            s = s.replace('\\\\', '\\')
            s = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

            return s

        # 其他 literal（数字、dict、list 等）
        try:
            return ast.literal_eval(arg_str)
        except Exception:
            return arg_str
