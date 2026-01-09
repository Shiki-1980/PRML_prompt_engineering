# ARC 推理系统 - 多策略推理引擎

这是一个针对 **Abstraction and Reasoning Corpus (ARC)** 视觉推理任务的智能求解系统，集成了多种先进的大语言模型推理策略。

## 🌟 项目特点

- **多策略求解**: 提供5种不同的推理方法，从基础到高级
- **自动验证**: 内置代码执行和逻辑验证机制
- **自我反思**: 实现Critic-Refiner架构，提升推理准确性
- **投票机制**: 通过自我一致性采样提高稳定性
- **模块化设计**: 易于扩展新的推理策略

## 📁 项目结构

```markdown
ARC-Solver/
├── main.py                    # 主程序入口
├── construct_prompt.py        # 提示词构建和解析
├── utils/
│   ├── code_for_reasoning.py  # 代码执行和验证模块
│   └── methods.py            # 各种推理方法实现
├── data/
│   └── val.jsonl             # ARC验证数据集
└── requirements.txt          # 依赖列表
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd ARC-Solver

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

在 `main.py`中设置您的API密钥(内置的api-key不可用)：

```python
API_KEY = 'your-api-key-here'
BASE_URL = "https://api.deepseek.com"  # 或其他API端点
MODEL_NAME = "deepseek-chat"           # 或其他模型
```

系统提供5种推理策略，可通过 `--method`参数选择：

## 使用方法

```bash
# 基础格式
python main.py --method [方法名] [其他参数]
# 查看所有方法
python main.py -h
```

### 1. **direct** (基础模式)

- 单次推理，直接输出结果
- 速度最快，适合简单任务

### 2. **sc** (自我一致性)

- 多次并行采样 + 多数投票机制
- 通过 `--sample`控制采样次数
- 提高结果的稳定性

### 3. **reflexion** (反思模式)

- Critic-Refiner 两阶段反思架构
- 自动识别逻辑错误并修正
- 适合复杂推理任务

### 4. **code** (代码生成模式)

- 生成Python代码实现转换规则
- 自动执行验证训练样本
- 支持调试和重试

### 5. **hybrid** (混合模式)

- 自我一致性 + 反思机制
- 先采样投票选出候选解，再进行逻辑校验

------

# ⚙️ 配置选项

## 数据范围选择

```bash
# 运行索引7-12的任务（7,8,9,10,11）
python main.py --data 7:12 --method reflexion
# 运行单个指定任务(Task3)
python main.py --data 3 --method hybrid
# 运行索引1,3,5的任务
python main.py --data 1,3,5 --method sc
# 运行索引3-5,8,10-12的任务
python main.py --data 3:5,8,10:12 --method reflexion
#运行所有任务
python main.py --data all --method direct
```

## 模型参数配置

### 基本API设置

```bash
# 指定模型名称
python main.py --model deepseek-chat --method direct --data 5
# 自定义API密钥和端点
python main.py --api_key your-key-here --base_url https://api.example.com --method direct --data 5
# 设置采样温度（控制随机性）
python main.py --temperature 0.8 --method direct --data 5
# 设置采样次数（仅对sc和hybrid有效）
python main.py --sample 5 --method sc --data 5
```

## 完整参数参考

| 参数            | 缩写 | 类型  | 默认值                   | 说明                                          |
| --------------- | ---- | ----- | ------------------------ | --------------------------------------------- |
| `--method`      | `-m` | str   | `direct`                 | 推理方法：direct, sc, reflexion, code, hybrid |
| `--data`        | `-d` | str   | `all`                    | 数据范围：数字、范围、列表或"all"             |
| `--task_id`     | `-i` | str   | 无                       | 指定任务ID运行单个任务                        |
| `--task_index`  | `-x` | int   | 无                       | 指定任务索引运行单个任务                      |
| `--sample`      | `-s` | int   | 3                        | 采样次数（仅sc/hybrid有效）                   |
| `--temperature` | `-t` | float | 1.0                      | 采样温度（0.0-2.0）                           |
| `--model`       | 无   | str   | deepseek-chat            | 模型名称                                      |
| `--api_key`     | 无   | str   | 内置默认                 | API密钥                                       |
| `--base_url`    | 无   | str   | https://api.deepseek.com | API端点                                       |
| `--data_path`   | 无   | str   | data/val.jsonl           | 数据文件路径                                  |