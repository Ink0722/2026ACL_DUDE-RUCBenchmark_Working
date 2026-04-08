import os
import time
import json
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import functools
from datetime import datetime

from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from types import MethodType

from src import hybrid_label_confidence_reward
from src import load_local_dataset
from src import make_conversation
from src import Local



class IntegratedTrainOptimize:
    def __init__(self, 
                 model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                 data_path="data/small_deception.json",
                 images_dir="data/images",
                 output_dir="Qwen2.5-VL-3B-Click-NewPlan3",
                 device="cpu",
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
        self.train_dataset = None
        self.test_dataset = None
        
        # 初始化模型和处理器
        self.model = None
        self.processor = None
        
        # 训练后的模型路径
        self.trained_model_path = None

        if use_api:
            self.API_KEY = api_key
            self.BASE_URL = base_url
        
    def load_data(self, test_size=0.2, seed=42):
        """加载和准备数据集"""
       
        print("Loading dataset...")
        self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False,Train=True)
        # self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False,Train=True, shuffle=shuffle, seed=seed)

        # Shuffle数据集（没有必要，split函数自己会分）
        # self.dataset = shuffle(self.dataset, seed)

        # 分割数据集
        split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        print("======================")
        print(split_dataset["train"][0])
        print("======================")
        print(split_dataset["test"][0])
        print("======================")
        self.train_dataset = split_dataset['train']
        print(f"LEN {len(self.train_dataset)}")
        self.test_dataset = split_dataset['test']
        print(f"LEN {len(self.test_dataset)}")

        # 格式化训练数据
        self.train_dataset = self.train_dataset.map(make_conversation)
        print("==========ATTENTION NOW============")
        print(self.train_dataset[0])
        print("======================")

        # 在训练模式下，将生成的 records 保存为 jsonl，文件名为 stage1_result + 时间戳.jsonl
        out_dir = os.path.join(os.getcwd(), "stage1_result")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        fname = f"stage1_result{ts}.jsonl"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as fw:
            for r in self.train_dataset:
                rec_copy = dict(r) if isinstance(r, dict) else {"record": r}
                rec_copy["status"] = None
                fw.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
        self.stage1_snapshot_path = out_path
        print(f"Saved dataset snapshot to {out_path}")

        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")
        
    def setup_model(self):
        """设置模型和处理器"""
        print("Setting up model and processor...")
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True, padding_side="left")
        
        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        
        # 设置LoRA配置
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 添加安全的批处理解码方法
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

        self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

    #可选在线模型轻量化启动
    def setup_cloud_model(self):
        self.model =  GLM(
                model_name="glm-4.6v",
                api_key=self.API_KEY,
                SYSTEM_PROMPT=""
            )

    def train_model(self, 
                   learning_rate=1e-5,
                   num_train_epochs=1,
                   per_device_train_batch_size=2,
                   num_generations=2,
                   max_completion_length=512,
                   max_prompt_length=8192,
                   logging_steps=10,
                   save_steps=10,
                   reward_type="combined"):
        """训练基座模型"""
        print("Starting model training...")
        
        # 生成带时间戳的单次训练记录文件（简单模式），并传入 run_ts
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorded_basename = f"recorded_samples_{ts}.jsonl"
        recorded_path = os.path.join(self.output_dir, recorded_basename)
        os.makedirs(self.output_dir, exist_ok=True)
        # 使用最小化修改：不要传入 functools.partial（没有 __name__），
        # 而是传入一个具名 wrapper，使 GRPOTrainer 能读取 __name__。
        # 捕获 snapshot 路径到闭包，确保序列化到 worker 时仍可访问（更稳健）
        snapshot = getattr(self, 'stage1_snapshot_path', None)
        def _reward_wrapper(*a, **kw):
            return hybrid_label_confidence_reward(*a, recorded_samples_path=recorded_path, snapshot_path=snapshot, run_ts=ts, **kw)

        # 保证有可读的名字
        try:
            _reward_wrapper.__name__ = getattr(hybrid_label_confidence_reward, "__name__", "hybrid_label_confidence_reward")
        except Exception:
            pass

        reward_funcs = [_reward_wrapper]
        
        # 配置训练参数
        training_args = GRPOConfig(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            remove_unused_columns=False,
            num_train_epochs=num_train_epochs,
            bf16=True,
            per_device_train_batch_size=per_device_train_batch_size,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
        )
        
        # 创建训练器
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=self.train_dataset,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model(self.output_dir)
        self.trained_model_path = self.output_dir
        
        print(f"Model training completed. Model saved to {self.output_dir}")
        
    def load_trained_model(self, model_path=None):
        """加载训练好的模型用于优化"""
        if model_path is None:
            model_path = self.trained_model_path
        
        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")
        
        print(f"Loading trained model from {model_path}...")
        
        # 创建本地Agent实例
        self.agent = Local(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            SYSTEM_PROMPT="",
            tools=[]
        )
        
        # 加载训练好的权重
        from peft import PeftModel
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.agent.model = PeftModel.from_pretrained(base_model, model_path)
        
        print("Trained model loaded successfully for optimization")
        
    
    def run_full_pipeline(self, 
                         train_params=None,
                         optimize_params=None):
        """运行完整的训练和优化流水线"""
        print("Starting full training and optimization pipeline...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 设置模型
        self.setup_model()
        
        # 3. 训练模型
        train_params = train_params or {}
        self.train_model(**train_params)
        

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


def demo_reward_types():
    """演示不同奖励类型的使用"""
    print("演示不同奖励类型的使用...")
    
    system = IntegratedTrainOptimize()
    system.load_data()
    system.setup_model()
    
    # 测试不同奖励类型
    reward_types = ['v2_reward']
    
    for reward_type in reward_types:
        print(f"\n=== 测试奖励类型: {reward_type} ===")
        
        # 只训练1个epoch作为演示
        try:
            system.train_model(
                num_train_epochs=1,
                per_device_train_batch_size=1,
                reward_type=reward_type,
                logging_steps=5,
                save_steps=10
            )
            print(f"✅ 奖励类型 '{reward_type}' 训练成功")
        except Exception as e:
            print(f"❌ 奖励类型 '{reward_type}' 训练失败: {e}")


if __name__ == "__main__":
    main()
    # demo_reward_types()