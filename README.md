# Qwen2-VL 多模态大模型微调实战

本项目是一个基于 Qwen2-VL-2B-Instruct 模型的多模态大模型微调练习，使用 COCO2014 图像描述数据集进行 LoRA 微调训练。

## 📋 项目概述

本项目实现了：
- 使用 transformers、peft 等框架进行模型微调
- 基于 COCO2014 图像描述数据集的多模态训练
- 集成 SwanLab 进行训练过程监控
- LoRA（Low-Rank Adaptation）微调技术实践

## 🛠️ 技术栈

- **模型**: Qwen2-VL-2B-Instruct
- **数据集**: COCO 2014 Caption（500张图像）
- **微调方法**: LoRA (Low-Rank Adaptation)
- **训练监控**: SwanLab
- **深度学习框架**: PyTorch + Transformers + PEFT

## 📁 项目结构

```
e:\LLM\
├── Qwen\                          # 模型文件目录
│   └── Qwen2-VL-2B-Instruct\     # 下载的预训练模型
├── coco_2014_caption\             # COCO2014图像数据
├── coco-2024-dataset.csv          # 处理后的数据集CSV文件
├── data_vl.json                   # 转换后的JSON格式训练数据
├── data2csv.py                    # 数据集下载和CSV转换脚本
├── csv2json.py                    # CSV到JSON格式转换脚本
├── load_model.py                  # 模型下载和加载脚本
├── train.py                       # 模型微调训练脚本
├── requirements.txt               # 项目依赖
└── README.md                      # 项目说明文档
```

## 🚀 快速开始

### 1. 环境配置

确保你的电脑上至少有一张英伟达显卡，并已安装好了CUDA环境。

```bash
# 创建conda虚拟环境
conda create -n llm python=3.11
conda activate llm

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载并处理COCO2014数据集
python data2csv.py

# 转换为训练格式
python csv2json.py
```

### 3. 模型下载

```bash
# 下载Qwen2-VL-2B-Instruct模型
python load_model.py
```

### 4. 开始训练

```bash
# 启动LoRA微调训练
python train.py
```

## 📊 数据集说明

**COCO 2014 Caption数据集**是Microsoft Common Objects in Context (COCO)数据集的一部分，主要用于图像描述任务。本项目使用其中的500张图像进行训练。

数据格式示例：
```json
[
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "请描述这张图片: <|vision_start|>图像文件路径<|vision_end|>"
      },
      {
        "from": "assistant",
        "value": "A modern kitchen done in white with a living room near by."
      }
    ]
  }
]
```

## 🔧 核心依赖

```txt
modelscope==1.18.0
transformers==4.46.2
sentencepiece==0.2.0
acceleratepeft==0.13.2
swanlab==0.4.5
datasets==2.18.0
pandas==2.0.3
qwen-vl-utils==0.0.8
numpy==1.24.3
```

## 📈 训练监控

本项目集成了 SwanLab 进行训练过程可视化监控，可以实时查看：
- 训练损失变化
- 学习率调度
- 模型性能指标
- 硬件资源使用情况

## 🎯 学习目标

通过本项目，你将学习到：
1. 多模态大模型的基本原理
2. LoRA微调技术的实际应用
3. 图像-文本数据的处理方法
4. 模型训练过程的监控和调优
5. 深度学习项目的完整流程

## 📚 参考资料

- [Qwen2-VL多模态大模型微调实战（完整代码）](https://blog.csdn.net/SoulmateY/article/details/143807035)
- [Qwen2-VL 官方文档](https://github.com/QwenLM/Qwen2-VL)
- [PEFT 官方文档](https://github.com/huggingface/peft)
- [SwanLab 官网](https://swanlab.cn)

## ⚠️ 注意事项

1. **硬件要求**: 建议使用至少8GB显存的GPU进行训练
2. **内存要求**: 建议系统内存不少于16GB
3. **存储空间**: 模型和数据集大约需要10GB存储空间
4. **训练时间**: 根据硬件配置，完整训练可能需要数小时

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目仅用于学习和研究目的。