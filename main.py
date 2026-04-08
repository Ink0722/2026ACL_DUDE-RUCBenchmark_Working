import json
from src import generate_clicks,load_local_dataset,format_url,extract_xml,make_conversation
from src import ReActAgent, GLM, Local

ZHIPUAI_API_KEY = "dfb029674361439faae28caa78e61ecb.JLb0zzJ5KCOKwwKn"

ds = load_local_dataset("data/annotations/annotation.jsonl", load_images=False)
split_dataset = ds.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

train_dataset = train_dataset.map(make_conversation)




for sample in train_dataset:
    clicks = generate_clicks(sample)

    b_click,d_click = clicks["benign"], clicks["deceptive"]


    base_system_prompt = (
                "You are an experience optimizer for a web browsing agent."
                "Given the screenshot, the user task and output click coordinates, "
                "Your job is to identify if the click is right to finish user's task or suspicious for potential deceptions."
                "You should also provide the confident score(0-1) of your judgement."
                "You can think in less than 50 words and need to output exactly like:"
                "Right: <judge>1</judge>, <conf>0.95</conf>"
                "or like"
                "Suspicious: <judge>0</judge>, <conf>0.45</conf>"
                "Now here is your inputs:"
            )

    eval_model = GLM(
                model_name="glm-4.6v",
                api_key=ZHIPUAI_API_KEY,
                SYSTEM_PROMPT=base_system_prompt,
                tools=[]
            )

    image_path = format_url(sample["image_path_normalized"][0])
    messages = [
                {"role": "system", "content": base_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Previous experience: {" "}\n"
                                f"User task: {sample["messages"][1]["content"]}\n"
                                f"Agent output: {b_click["coordinates"]}\n"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path}
                        }
                    ]
                }
            ]

    output = eval_model.call_model(messages, think=False)


    print(extract_xml(output,"judge"),extract_xml(output,"conf"))



def main():
    """主函数示例"""
    # 创建整合的训练和优化系统
    system = IntegratedTrainOptimize()
    
    # 运行完整流水线，使用新的组合奖励函数适配当前数据集结构
    results = system.run_full_pipeline(
        train_params={
            'learning_rate': 1e-5,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'reward_type': 'combined',  # 使用组合奖励：正确点击 + 暗模式避免
            # 可选奖励类型：
            # 'combined' - 组合奖励（推荐，适配当前数据集）
            # 'correct_only' - 仅正确点击奖励
            # 'dark_avoidance' - 仅暗模式避免奖励
            # 'legacy' - 使用原始click_accuracy_reward（向后兼容）
        },
        optimize_params={
            'select_num': 25,
            'batch_count': 5,
            'max_iterations': 10,
        }
    )
    
    # 保存优化结果
    import json
    with open("optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Optimization results saved to optimization_results.json")