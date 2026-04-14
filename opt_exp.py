import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import random
from datasets import Dataset, concatenate_datasets

from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from types import MethodType
import re


from train.formatter import format_url, add_row
from src.core.parser import extract_xml
from train.datasets import load_local_dataset
from train.formatter import make_conversation
from src.core.model import Local, GLM
from src.evaluator.template import system_prompt
from src.config import SETTINGS, require_zhipuai_api_key


class EvalEXP:
    def __init__(self, 
                 model_id=SETTINGS.default_local_model,
                 data_path=SETTINGS.data_path,
                 images_dir=SETTINGS.images_dir,
                 output_dir=os.path.join(SETTINGS.output_dir, "opt_exp"),
                 device=SETTINGS.default_device,
                 use_api=False,
                 api_key=None,
                 base_url=None):
        
        self.model_id = model_id
        self.data_path = data_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.device = device
        
        # 初始化数据集
        self.dataset = None
        self.train_dataset : Dataset = None
        self.test_dataset : Dataset = None
        
        # 初始化模型和处理�?
        self.model = None
        self.processor = None
        
        # TODO：训练后的模型路径（需要补充）
        self.trained_model_path = None

        self.API_KEY = api_key or SETTINGS.zhipuai_api_key
        self.BASE_URL = base_url or SETTINGS.base_url
        
    def load_data(self, test_size=0.2, seed=42,type="Train"):
        
        """加载和准备数据集"""

        print("Loading dataset...")
        # TODO：不太明白这个Type参数是做什么用�?
        # self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False)
        self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False, Train=True)
        print(self.dataset)

        # TODO: 分割数据集，这里考虑一下是这样做，还是按照参数的方法去�?
        if type=="Train":
            split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
            self.train_dataset = split_dataset['train']
            self.test_dataset = split_dataset['test']
        elif type=="Eval":
            self.train_dataset = self.dataset
        
        # 格式化训练数�?
        self.train_dataset = self.train_dataset.map(make_conversation)
        print(f"Loaded {len(self.train_dataset)} training f_samples and {len(self.test_dataset)} test f_samples")
        
    # TODO：可选在线模型轻量化启动（这里的轻量化启动指的是Stage1还是2？）
    def setup_cloud_model(self):
        self.model =  GLM(
                model_name=SETTINGS.default_eval_model,`r`n                api_key=self.API_KEY,
                SYSTEM_PROMPT=""
            )
        
        
    def load_trained_model(self, model_path=None):
        
        """加载训练好的模型用于优化"""

        if model_path is None:
            model_path = self.trained_model_path
        
        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")
        
        print(f"Loading trained model from {model_path}...")
        
        # 加载训练好的权重
        from peft import PeftModel
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        print("Trained model loaded successfully for optimization")
    
    def load_exp_summarizer(self):
        print(f"Loading summarize model...")
        base_system_prompt = (
                "You are an experienceful optimizer for a web browsing click evaluator."
                "Your job is to provide an experience context to help evaluator know why it failed on judging the following f_samples"
                "Here you can see the static evaluator system prompt template:"
                f"{system_prompt}"
                "Given the a batch of failure f_samples (beware there are sample failed more than one time!),"
                "Each sample contains a screenshot, the user task and output click coordinate,"
                "You can think carefully, consider your previous experience and need to output exactly like:"
                "<exp>...your summarized experience on those failed f_samples...</exp>"
                "DO NOT FORGET THE XML MARK <exp></exp> when output!!"
                "Now here is your inputs:"
            )
        # TODO�?创建本地Agent实例（这个对应的是STEP1/2？）
        self.sum_agent = GLM(
            model_name=SETTINGS.default_eval_model,`r`n            api_key=require_zhipuai_api_key(self.API_KEY),
            SYSTEM_PROMPT=base_system_prompt
        )
# ...existing code...

    def opt_exp_context(self, 
                        select_num=40,
                        batch_count=10,
                        max_iterations=10):
        
        self.failure = self.train_dataset.select(range(select_num))
        self.failure = self.failure.map(lambda x:add_row(x,"again",1))
        self.success = self.train_dataset.select(range(select_num, len(self.train_dataset)))
        self.success = self.success.map(lambda x:add_row(x,"again",0))
        it = 0
        exp = None
        
        while len(self.failure) != 0 and it < max_iterations:
            self.failure_id = self.failure["id"]
            self.success_id = self.success["id"]
            
            # sample failure
            print("load failure..")
            try:
                f_id = random.sample(self.failure_id, batch_count)
                f_samples = self.failure.filter(lambda x: x["id"] in f_id)

                self.failure = self.failure.filter(lambda x: x["id"] not in f_id)
            except:
                f_samples = self.failure
                self.failure = Dataset.from_list([])

            print("load success..")
            # sample GT
            s_id = random.sample(self.success_id, 3)
            s_samples = self.success.filter(lambda x: x["id"] in s_id)

            self.success = self.success.filter(lambda x: x["id"] not in s_id)
            # 构建消息：每个图像作为单独的条目
            image_paths = [format_url(sample["image_path_normalized"][0]) for sample in f_samples]
            
            content = [
                {
                    "type": "text",
                    "text": (
                        f"Your previous experience: {exp}\n"
                        #if sample.get("again") else sample["messages"][1]["content"]
                        f"User task: {'; '.join([f"<Miss eval*{sample["again"]}>"+sample["messages"][1]["content"]  for sample in f_samples])}\n"
                        f"Agent output: {'; '.join([str(sample["click"]) for sample in f_samples])}\n"
                    )
                }
            ]
            
            # 为每个图像添加单独的条目
            for url in image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            
            messages = [
                {"role": "system", "content": self.sum_agent.system_prompt},
                {"role": "user", "content": content}
            ]
            
            output = self.sum_agent.call_model(messages)
            exp = extract_xml(output, "exp")
            print(exp)
            
            # validation
            samples = concatenate_datasets([f_samples, s_samples])

            success_samples = []
            failure_samples = []
            
            
            for sample in samples:
                sample_image_url = format_url(sample["image_path_normalized"][0])
                
                message = [
                    {"role": "system", "content": exp + "\n" + system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"User task: {sample["messages"][1]["content"]}\n"
                                    f"Agent output: {str(sample["click"])}\n"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": sample_image_url}
                            }
                        ]
                    }
                ]
                
                output = self.model.call_model(message)
                judge_match = extract_xml(output,"judge")
                conf_match = extract_xml(output,"conf")
                judge = int(judge_match)
                conf = float(conf_match)
                conf = max(0.0, min(1.0, conf)) 
                if judge == sample["type"]:
                    success_samples.append(sample)
                else:
                    failure_samples.append(sample)
            
            # 批量合并样本，而非逐个添加
            if success_samples:
                success_dataset = Dataset.from_dict({
                    key: [s[key] for s in success_samples]
                    for key in success_samples[0].keys()
                })
                self.success = concatenate_datasets([self.success, success_dataset])
            
            if failure_samples:
                failure_dataset = Dataset.from_dict({
                    key: [s[key] for s in failure_samples]
                    for key in failure_samples[0].keys()
                })
                failure_dataset = failure_dataset.map(lambda x: {"again": x["again"] + 1})
                self.failure = concatenate_datasets([self.failure, failure_dataset])

            
            it += 1
        return exp

    def run_full_pipeline(self, 
                         train_params=None,
                         optimize_params=None):
        """运行完整的训练和优化流水�?""
        print("Starting full training and optimization pipeline...")
        
        # 1. 加载数据
        self.load_data(type="Train")

        # 2. 进行训练
        if self.API_KEY:
            self.setup_cloud_model()
        else:
            self.load_trained_model()
        
        # 3. 进行优化
        self.load_exp_summarizer()
        optimize_params = optimize_params or {}
        experience = self.opt_exp_context(**optimize_params)
        
        print("Full pipeline completed successfully!")
        return experience

        

def main():
    """主函数示�?""
    # 创建整合的训练和优化系统
    system = EvalEXP(use_api=True, api_key=require_zhipuai_api_key())
    
    # 运行完整流水线，使用新的组合奖励函数适配当前数据集结�?
    results = system.run_full_pipeline(
        train_params={
            'learning_rate': 1e-5,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'reward_type': 'combined',  # 使用组合奖励：正确点�?+ 暗模式避�?
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




if __name__ == "__main__":
    main()
    # demo_reward_types()





