# Qwen2-VL Fine-tuning Project

本项目用于对Qwen2-VL模型进行LoRA微调，实现图像描述生成任务。

## 项目结构

```
Qwen2-VL_fine-tuning/
├── data/                    # 数据相关
│   ├── raw/                # 原始数据
│   │   └── coco_2014_caption/  # COCO 2014图像数据
│   ├── processed/          # 处理后的数据
│   │   ├── coco-2024-dataset.csv
│   │   ├── data_vl.json
│   │   ├── data_vl_train.json
│   │   └── data_vl_test.json
│   └── sample_images/      # 示例图片
│       └── sample_image1.png
├── scripts/                # 脚本文件
│   ├── data_processing/    # 数据处理脚本
│   │   ├── data2csv.py     # 下载数据并转换为CSV
│   │   └── csv2json.py     # 将CSV转换为训练用JSON格式
│   ├── training/           # 训练相关
│   │   └── train.py        # LoRA微调训练脚本
│   └── inference/          # 推理相关
│       └── lora_inference.py  # 使用微调模型进行推理
├── models/                 # 模型文件
│   └── Qwen/
│       └── Qwen2-VL-2B-Instruct/  # 预训练模型
├── output/                 # 输出结果
│   └── Qwen2-VL-2B/
│       └── checkpoint-62/  # LoRA微调后的模型权重
├── configs/                # 配置文件
├── utils/                  # 工具函数
│   └── load_model.py       # 模型加载工具
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明
└── .gitignore             # Git忽略文件
```

## 使用方法

### 1. 数据处理
```bash
# 进入数据处理目录
cd scripts/data_processing

# 下载COCO数据并转换为CSV
python data2csv.py

# 将CSV转换为训练用JSON格式
python csv2json.py
```

### 2. 模型训练
```bash
# 进入训练目录
cd scripts/training

# 开始LoRA微调训练
python train.py
```

### 3. 模型推理
```bash
# 进入推理目录
cd scripts/inference

# 使用微调后的模型进行推理
python lora_inference.py
```

## 注意事项

- 确保已安装所有依赖包：`pip install -r requirements.txt`
- 训练需要GPU支持，建议使用CUDA环境
- 模型文件较大，首次运行会自动下载
- 所有路径引用已更新为相对于新目录结构的路径