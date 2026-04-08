import json
import re

def parse_agent_action(output: str) -> dict:
    """
    解析 agent 输出的 Docstring
    output 示例:
    Action: <action_name>
    Action Input: <input_parameters_in_json_format>
    """
    action_pattern = r'Action:\s*(\w+)'
    input_pattern = r'Action Input:\s*(\{.*\})'

    action_match = re.search(action_pattern, output)
    input_match = re.search(input_pattern, output, re.DOTALL)

    if action_match and input_match:
        action_name = action_match.group(1)
        input_json_str = input_match.group(1)

        try:
            input_params = json.loads(input_json_str)
        except json.JSONDecodeError:
            input_params = {}

        return {
            "action": action_name,
            "input": input_params
        }
    else:
        return {
            "action": None,
            "input": {}
        }

def parse_agent_output(output: str) ->tuple:
    """
    识别输出中的click坐标(x,y)
    """
    click_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    match = re.search(click_pattern, output)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x,y)
    else:
        return (0,0)
    
def extract_xml(output: str,tag:str):
    match = re.search(f"<{tag}>(.*?)</{tag}>", output,re.DOTALL)
    return match.group(1) if match else ""