# LLaDA-Table-SFT

> **Table Reasoning Fine-Tuning based on LLaDA**

这是一个基于 **dLLM** 框架进行二次开发的表格推理（Table Reasoning）微调项目。本项目主要针对 **LLaDA-8B** 模型进行 SFT (Supervised Fine-Tuning)，通过引入表格数据，使其能够更好地理解表格结构并进行逻辑推理。

🔗 **Upstream Repository:** 本项目基于 [dLLM (Simple Diffusion Language Modeling)](https://github.com/ZHZisZZ/dllm)修改。

如需查看原始框架的详细文档或底层实现，请访问原仓库。

## 1. 环境准备 (Setup)

### Step 1: 创建并激活新环境

建议使用 Python 3.10 环境：

```Bash
conda create -n dllm python=3.10 -y
conda activate dllm
```

### Step 2: 安装依赖

安装 PyTorch 及项目依赖。

```Bash
pip install -r requirements.txt
```

### Step 3: 安装本项目 (dLLM)

以编辑模式安装，方便调试代码：

```Bash
pip install -e .
```

## 2. 数据准备 (Data Preparation)

```Bash
python download_data.py
```

## 3. SFT