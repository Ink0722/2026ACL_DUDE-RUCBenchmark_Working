"""Stage-1 风格的批量/单样本推理脚本

流程（对应图片上方 STAGE 1）:
 1. 从混合数据集读取样本（或单样本输入）
 2. 基于规则生成 benign/deceptive 点击（若需）
 3. 用 Evaluator（Local/基座或训练后模型）对每个样本做判断，输出 judge/conf
 4. 比较 judge 与 GT label，计算奖励（正/负，按置信度加权），并保存记录

用法示例：
  python stage1_inference.py --mode finetuned --adapter-dir Qwen2.5-VL-3B-Click \
    --ann-path data/Deception.json --images-dir data/images --limit 100 --save results_stage1.jsonl
"""
import argparse
import json
import torch
import os
from typing import List
from datetime import datetime

from src import Local
from src.utils import load_local_dataset
from src.core import label_confidence_reward, extract_xml

# 常量配置
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGES_DIR = "data/images"
DEFAULT_ANN_PATH = "data/small_deception.json"
DEFAULT_ADAPTER_DIR = "Qwen2.5-VL-3B-Click-NewPlan3"
DEFAULT_OUT_DIR = "results"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_agent(model_id: str, adapter_dir: str | None, mode: str, system_prompt: str | None = None):
    model_path = adapter_dir if mode == "finetuned" and adapter_dir and os.path.exists(adapter_dir) else None
    agent = Local(model_name=model_id, SYSTEM_PROMPT=system_prompt or "", tools=[], model_path=model_path)
    if mode == "finetuned" and adapter_dir and model_path is None:
        try:
            agent.load_model_from_path(adapter_dir)
        except Exception:
            pass
    return agent


def run_on_dataset(agent: Local, ann_path: str, images_dir: str, limit: int | None = None, out_dir: str | None = None, test_size=0.2, seed=42):
    
    
    initds = load_local_dataset(ann_path, images_dir, load_images=False, Train=True)
     # 仅看测试集结果
    split_dataset = initds.train_test_split(test_size=test_size, seed=seed)
    print("======================")
    print(split_dataset["test"][0])
    print("======================")
    ds = split_dataset['test']

    results: List[dict] = []
    count = 0

    for sample in ds:
        if limit and count >= limit:
            break
        # sample 中按 load_local_dataset 的 Train=True 已被扩展为包含 'click' 与 'type' 字段，以及 'messages'
        messages = sample.get("messages")
        # 规范化 messages：某些记录里的 message.content 可能是字符串，
        # processor.apply_chat_template 期望 content 是 list of dicts
        def normalize_messages(msgs):
            out_msgs = []
            if not isinstance(msgs, list):
                return msgs
            for m in msgs:
                try:
                    role = m.get("role", "user")
                    content = m.get("content")
                except Exception:
                    # 如果 m 不是 dict，跳过
                    continue

                norm_content = []
                if isinstance(content, str):
                    norm_content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    for it in content:
                        if isinstance(it, dict):
                            # 保持已有结构
                            norm_content.append(it)
                        elif isinstance(it, str):
                            norm_content.append({"type": "text", "text": it})
                        else:
                            # 最后兜底转为字符串
                            norm_content.append({"type": "text", "text": str(it)})
                elif isinstance(content, dict):
                    # 将单个 dict 包装成列表
                    norm_content = [content]
                else:
                    norm_content = [{"type": "text", "gen_text": str(content)}]

                out_msgs.append({"role": role, "content": norm_content})
            return out_msgs

        messages = normalize_messages(messages)
        
        # 注入图片到 messages 中，确保模型能看到图
        image_paths = sample.get("image_path_normalized")
        if image_paths and len(image_paths) > 0:
            for msg in messages:
                if msg["role"] == "user":
                    if isinstance(msg["content"], list):
                        # 将图片添加到用户消息中
                        msg["content"].append({"type": "image", "image": image_paths[0]})
                    break

        expected_type = sample.get("gen_type")  # 1 benign -> expect judge=1; 0 deceptive -> expect judge=0

        if messages is None:
            print("⚠️ sample missing messages field, skipping")
            continue

        # 调用 evaluator（捕获单样本异常，继续执行）
        try:
            out = agent.call_model(messages)
        except Exception as e:
            out = f"Error: {e}"

        # 解析 judge/conf
        try:
            judge_str = extract_xml(out, "judge")
        except Exception:
            judge_str = ""
        try:
            conf_str = extract_xml(out, "conf")
        except Exception:
            conf_str = ""

        # 尝试转换类型
        parsed_judge = None
        try:
            if judge_str != "":
                parsed_judge = int(judge_str)
        except Exception:
            parsed_judge = None

        parsed_conf = None
        try:
            if conf_str != "":
                parsed_conf = float(conf_str)
        except Exception:
            parsed_conf = None

        # 计算 reward（复用项目函数）
        rewards = label_confidence_reward([out], type=[expected_type])
        reward = rewards[0] if rewards else None

        rec = {
            "model_output": str(out),
            "judge": parsed_judge,
            "conf": parsed_conf,
            "correct_step": 0 if (reward is not None and reward > 0) else None
        }

        # Minimal change: 保存该 sample 的全部原始字段，同时保留现有的 rec 字段。
        merged = dict(sample) if isinstance(sample, dict) else {}
        merged.update(rec)

        results.append(merged)
        count += 1

        if count % 50 == 0:
            print(f"Processed {count} samples")

    # 统计汇总
    total = len(results)
    parsed = [r for r in results if r.get("judge") is not None and r.get("conf") is not None]
    parsed_count = len(parsed)
    # positive: 使用 new `correct_step` 字段判断（0 表示预测正确）
    pos = sum(1 for r in results if r.get("correct_step") == 0)
    # negative: 在已解析(parsed)样本中，correct_step 为 None 表示预测错误
    neg = sum(1 for r in parsed if r.get("correct_step") is None)
    parse_failures = sum(1 for r in results if r.get("judge") is None or r.get("conf") is None)

    avg_conf_overall = None
    if parsed_count > 0:
        avg_conf_overall = sum((r.get("conf") or 0.0) for r in parsed) / parsed_count

    # avg conf for matches / mismatches
    matches = [r for r in parsed if r.get("correct_step") == 0]
    mismatches = [r for r in parsed if r.get("correct_step") is None]
    avg_conf_match = sum((r.get("conf") or 0.0) for r in matches) / len(matches) if matches else None
    avg_conf_mismatch = sum((r.get("conf") or 0.0) for r in mismatches) / len(mismatches) if mismatches else None

    pass_rate = pos / total if total > 0 else 0.0

    # 计算额外指标：DSR, TCR, Fatal Error Rate（只基于已解析样本，不计入 parse failures）
    def _safe_div(n, d):
        return (n / d) if d > 0 else None

    parsed_by_expected = {}
    for L in (1, -1, 0):
        parsed_by_expected[L] = [r for r in parsed if r.get("gen_type") == L]

    denom_neg1 = len(parsed_by_expected.get(-1, []))
    dsr_num = sum(1 for r in parsed_by_expected.get(-1, []) if r.get("judge") in (-1, 0))
    dsr = _safe_div(dsr_num, denom_neg1)

    denom_pos1 = len(parsed_by_expected.get(1, []))
    tcr_num = sum(1 for r in parsed_by_expected.get(1, []) if r.get("judge") == 1)
    tcr = _safe_div(tcr_num, denom_pos1)

    fatal_num = sum(1 for r in parsed_by_expected.get(-1, []) if r.get("judge") == 1)
    fatal = _safe_div(fatal_num, denom_neg1)

    summary = {
        "total": total,
        "parsed_count": parsed_count,
        "parse_failures": parse_failures,
        "positive": pos,
        "negative": neg,
        "pass_rate": pass_rate,
        "avg_conf_overall": avg_conf_overall,
        "avg_conf_match": avg_conf_match,
        "avg_conf_mismatch": avg_conf_mismatch,
        "DSR": dsr,
        "TCR": tcr,
        "Fatal_Error_Rate": fatal,
    }

    # 保存结果为 jsonl，并在最后追加 summary 行（便于后处理）
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"stage1_results_{ts}.jsonl"
        save_path = os.path.join(out_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            # summary 行标记为 __summary__ = True
            f.write(json.dumps({"__summary__": True, "summary": summary}, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} results + summary to {save_path}")

    # 控制台输出汇总的简短结果字符串
    summary_str = (
        f"Processed={total}, Passed={pos}, Failed={neg}, ParseFailures={parse_failures}, "
        f"PassRate={pass_rate:.3f}, AvgConf={avg_conf_overall if avg_conf_overall is None else f'{avg_conf_overall:.3f}'}"
    )
    print("=== Summary ===")
    print(summary_str)

    return results, summary


def main():
    parser = argparse.ArgumentParser()
    # 简化参数：保留常改变的选项，隐藏或使用内置默认值
    parser.add_argument("--mode", choices=["base", "finetuned"], default="finetuned", help="使用基座模型或训练后模型")
    parser.add_argument("--limit", type=int, default=None, help="处理样本上限，默认全部处理")
    args = parser.parse_args()

    # adapter 使用脚本顶部默认路径（若需改请修改常量 DEFAULT_ADAPTER_DIR）
    agent = load_agent(MODEL_ID, DEFAULT_ADAPTER_DIR, args.mode)
    out_dir = DEFAULT_OUT_DIR

    # 尝试把模型移动到目标设备（GPU/CPU）以加速推理
    device = DEFAULT_DEVICE
    try:
        if hasattr(agent, 'model') and agent.model is not None:
            agent.model.to(device)
            print(f"Moved model to device: {device}")
    except Exception as e:
        print(f"⚠️  Failed to move model to {device}: {e}")

    # Monkey-patch processor.apply_chat_template to ensure returned tensors are on model device
    try:
        if hasattr(agent, 'processor') and hasattr(agent.processor, 'apply_chat_template') and hasattr(agent, 'model') and agent.model is not None:
            from types import MethodType

            orig_apply = agent.processor.apply_chat_template

            def apply_and_move(self, *args, **kwargs):
                # Ensure we request tensors
                kwargs.setdefault('return_tensors', 'pt')
                inputs = orig_apply(*args, **kwargs)
                # move tensors to model device
                try:
                    model_device = None
                    # prefer model parameters' device
                    try:
                        model_device = next(agent.model.parameters()).device
                    except Exception:
                        # fallback to attribute
                        model_device = getattr(agent.model, 'device', None)

                    if isinstance(inputs, dict) and model_device is not None:
                        moved = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
                        return moved
                except Exception:
                    pass
                return inputs

            agent.processor.apply_chat_template = MethodType(apply_and_move, agent.processor)
            print("Patched processor.apply_chat_template to move tensors to model device")
    except Exception as e:
        print(f"⚠️  Failed to monkey-patch processor: {e}")
    
    
    # 批量推理（基于数据集）
    results, summary = run_on_dataset(agent, DEFAULT_ANN_PATH, IMAGES_DIR, limit=args.limit, out_dir=out_dir)
    # 额外的简短统计（已经在 run_on_dataset 中打印过详细 summary）
    print(f"Total processed: {summary.get('total')}, positive: {summary.get('positive')}, negative: {summary.get('negative')}")


if __name__ == "__main__":
    main()
