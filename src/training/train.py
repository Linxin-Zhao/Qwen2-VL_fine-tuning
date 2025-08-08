import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os
import glob


# 配置图像目录路径 - 使用正确的路径
IMAGE_BASE_DIRS = [
    "../data/raw/coco_2014_caption",  # 项目目录下的coco_2014_caption文件夹
    "E:\\LLM\\Qwen2-VL_fine-tuning\\Qwen2-VL_fine-tuning\\coco_2014_caption",  # 完整路径
    "E:\\LLM\\coco_2014_caption",  # 原始错误路径（备用）
]

def find_image_file(original_path):
    """
    根据原始路径查找实际存在的图像文件
    """
    # 提取文件名
    filename = os.path.basename(original_path)
    
    # 在多个可能的目录中查找
    for base_dir in IMAGE_BASE_DIRS:
        if os.path.exists(base_dir):
            # 直接路径匹配
            full_path = os.path.join(base_dir, filename)
            if os.path.exists(full_path):
                return full_path
            
            # 递归查找
            pattern = os.path.join(base_dir, "**", filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
    
    # 如果都找不到，返回None
    return None

def process_func(example):
    """
    将数据集进行预处理
    """
    try:
        MAX_LENGTH = 8192
        input_ids, attention_mask, labels = [], [], []
        conversation = example["conversations"]
        input_content = conversation[0]["value"]
        output_content = conversation[1]["value"]
        original_file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取原始图像路径
        
        # 查找实际存在的图像文件
        file_path = find_image_file(original_file_path)
        
        if file_path is None:
            print(f"Warning: Image file not found for: {original_file_path}, skipping...")
            return None
        
        # 转换为绝对路径
        file_path = os.path.abspath(file_path)
        print(f"Using image: {file_path}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{file_path}",
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                    {"type": "text", "text": "COCO Yes:"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
        instruction = inputs

        response = tokenizer(f"{output_content}", add_special_tokens=False)


        input_ids = (
                instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
        )

        attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [tokenizer.pad_token_id]
        )
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
        inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}
    
    except Exception as e:
        print(f"Error processing example: {e}")
        print(f"Problematic file path: {original_file_path if 'original_file_path' in locals() else 'Unknown'}")
        return None


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# 在modelscope上下载Qwen2-VL模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="../models/", revision="master")

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("../models/Qwen/Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("../models/Qwen/Qwen/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained("../models/Qwen/Qwen/Qwen2-VL-2B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
train_json_path = "../data/processed/data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

with open("../data/processed/data_vl_train.json", "w") as f:
    json.dump(train_data, f)

with open("../data/processed/data_vl_test.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("../data/processed/data_vl_train.json")
# 先映射处理函数，然后过滤掉None值
print("Processing training dataset...")
print(f"Checking image directories:")
for dir_path in IMAGE_BASE_DIRS:
    if os.path.exists(dir_path):
        print(f"  ✓ {dir_path} (exists)")
        # 显示目录中的图片数量
        image_files = glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.png"))
        print(f"    Found {len(image_files)} image files")
    else:
        print(f"  ✗ {dir_path} (not found)")

train_dataset_raw = train_ds.map(process_func, remove_columns=train_ds.column_names)
# 过滤掉处理失败的样本（返回None的样本）
train_dataset = train_dataset_raw.filter(lambda x: x is not None and all(v is not None for v in x.values()))
print(f"Original dataset size: {len(train_ds)}")
print(f"Filtered dataset size: {len(train_dataset)}")

if len(train_dataset) == 0:
    print("Error: No valid training samples found. Please check your image paths.")
    exit(1)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="../output/Qwen2-VL-2B",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)
        
# 设置SwanLab回调 (disabled for now)
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-finetune",
    experiment_name="qwen2-vl-coco2014",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "COCO Yes: ",
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2-VL-2B/checkpoint-62", config=val_config)

# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    # 获取原始图像路径
    original_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    # 查找实际存在的图像文件
    actual_image_path = find_image_file(original_image_path)
    
    if actual_image_path is None:
        print(f"Warning: Test image not found: {original_image_path}, skipping...")
        continue
    
    # 转换为绝对路径
    actual_image_path = os.path.abspath(actual_image_path)
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": actual_image_path
            },
            {
            "type": "text",
            "text": "COCO Yes:"
            }
        ]}]

    try:
        response = predict(messages, val_peft_model)
        messages.append({"role": "assistant", "content": f"{response}"})
        print(messages[-1])
        test_image_list.append(swanlab.Image(actual_image_path, caption=response))
    except Exception as e:
        print(f"Error predicting for {actual_image_path}: {e}")
        continue

if test_image_list:
    swanlab.log({"Prediction": test_image_list})
else:
    print("Warning: No test images were successfully processed.")

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
