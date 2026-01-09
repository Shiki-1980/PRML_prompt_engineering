# 这个文件的作用：在 ARC 的 jsonl 验证集上，串起完整的评测流程：
# 1）读取 jsonl 数据
# 2）对每个任务调用 construct_prompt 得到 prompt
# 3）调用大模型
# 4）用 parse_output 解析模型输出
# 5）统计有多少完全匹配 ground truth 并计算 accuracy

import ast
import os
import json
import time
from utils.code_for_reasoning import *
from openai import OpenAI
from utils.methods import *
from construct_prompt import *


# API_KEY='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'
API_KEY='sk-1e8eb86fea834022b8a231dba794b00d'
# MODEL_NAME = "nex-agi/deepseek-v3.1-nex-n1:free"
DATA_PATH = "data/val.jsonl"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_jsonl(path):
    """
    功能：
        从给定的 jsonl 文件中读取所有样本，并返回一个列表，每个元素是一个任务字典。
        每一行对应一个 ARC 任务（例如包含 "train" / "test" 等字段）。

    输入参数：
        path: 字符串形式的文件路径，例如 "val.jsonl"。

    返回值：
        data: 列表（list），其中每个元素是一个字典（dict），表示一个 ARC 任务。
              例如 data[i] = d_i，其中 d_i 可以直接传给 construct_prompt(d_i) 使用。
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        # 如果是标准 json 数组格式
        if path.endswith('.json'):
            data = json.load(f)
        else:
            # 如果是 jsonl 格式
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def check_accuracy(predictions, ground_truths):
    """
    功能：
        计算模型预测结果与 ground truth 之间的“完全匹配”准确率。
        完全匹配指：预测网格与真实网格在尺寸和每个元素上都完全相同。

    输入参数：
        predictions: 列表（list）
                     每个元素是模型预测的输出网格（通常是一个二维列表，如 [[0,1],[1,0],...]）。
        ground_truths: 列表（list）
                       每个元素是对应样本的真实输出网格（二维列表）。

    返回值：
        accuracy: 浮点数（float），表示完全匹配的比例
    """
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        # 比较两个二维列表是否完全一致
        if pred == gt:
            correct += 1
    
    return correct / len(ground_truths)

def load_tasks_custom(path, data_range=None):
    """
    path: 数据文件路径
    data_range: 
        - None: 加载全部
        - int (如 5): 加载前5个
        - tuple (如 (7, 12)): 加载索引从 7 到 11 的任务
        - list (如 [7, 11, 12]): 只加载索引为 7, 11, 12 的任务
    """
    all_tasks = load_jsonl(path)
    
    if data_range is None:
        return all_tasks
    
    if isinstance(data_range, int):
        return all_tasks[:data_range]
    
    if isinstance(data_range, tuple):
        start, end = data_range
        return all_tasks[start:end]
    
    if isinstance(data_range, list):
        # 根据索引列表提取，并加入异常检查防止越界
        selected_tasks = []
        for idx in data_range:
            if 0 <= idx < len(all_tasks):
                selected_tasks.append(all_tasks[idx])
            else:
                print(f"警告：索引 {idx} 超出数据范围，已跳过。")
        return selected_tasks

    return all_tasks


def speak_and_listen(messages, model_name, temperature=1.0):
    """
    功能：
        调用大语言模型 API，将 messages 作为对话输入，返回模型生成的文本回答。

        注意：
        - messages 通常是一个符合 OpenAI / 其他厂商接口格式的列表，
          由 construct_prompt(d) 生成，例如：
          [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
          ]
        - 本函数只负责“发送请求 + 接收模型回答”，不做解析。

    输入参数：
        messages: 列表（list），对话内容，由 construct_prompt(d) 返回。
        model_name: 字符串（str），要调用的模型名称，例如 "gpt-4o-mini"。
        temperature: 浮点数（float），采样温度，控制随机性，默认 0.0。

    返回值：
        reply_text: 字符串（str），表示模型的主回答文本内容。
                    之后会被交给 parse_output(reply_text) 进行网格解析。
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=8192,      
            stream=False          
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[API 错误]: {e}")
        return ""


def print_report(results_detail, accuracy, tasks_count, model_name, columns=None):
    """
    动态打印评测报告
    
    参数:
    results_detail (list): 包含每项结果字典的列表
    accuracy (float): 总准确率
    tasks_count (int): 任务总数
    model_name (str): 模型名称
    columns (list): 需要额外显示的列名，如 ['status', 'vote_count']
    """
    if columns is None:
        columns = [] 

    col_map = {
        "index": ("Idx", 6),
        "task_id": ("Task ID", 15),
        "status": ("Process Status", 30), # 宽度调大一点
        "result": ("Result", 10)
    }

    display_cols = ["index", "task_id"] + columns + ["result"]
    
    header_parts = [f"{col_map.get(c, (c, 12))[0]:<{col_map.get(c, (c, 12))[1]}}" for c in display_cols]
    header_str = " | ".join(header_parts)
    print("\n" + "=" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for res in results_detail:
        row = []
        for col in display_cols:
            width = col_map.get(col, (col, 12))[1]
            if col == "result":
                val = "PASS" if res['is_correct'] else "FAIL"
            else:
                val = str(res.get(col, "N/A"))
            row.append(f"{val:<{width}}")
        print(" | ".join(row))

    print("=" * len(header_str))
    print(f"MODEL: {model_name} | TOTAL: {tasks_count} | ACC: {accuracy:.2%}")

def main(data_range=None, method="direct"):
    print(f"开始评测模型: {MODEL_NAME} | 使用方法: {method}")
    
    # 1. 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到数据文件 {DATA_PATH}")
        return

    tasks = load_tasks_custom(DATA_PATH, data_range)
    all_predictions = []
    all_ground_truths = []
    results_detail = []

    # 2. 遍历任务
    for i, d in enumerate(tasks):
        actual_index = data_range[i] if isinstance(data_range, list) else i
        task_id = d.get('id', f"Task_{actual_index}")
        print(f"正在处理 [{i+1}/{len(tasks)}] {task_id}...", end=" ", flush=True)
        
        gt_grid = d['test'][0].get('output', [])

        # --- 核心调用：只需一行 ---
        # 你可以通过 kwargs 传入 sample_count 等参数
        predicted_grid, process_status = solve_task(method, d, MODEL_NAME, sample_count=3)
        # -----------------------

        all_predictions.append(predicted_grid)
        all_ground_truths.append(gt_grid)
        
        is_correct = (predicted_grid == gt_grid)
        status_icon = "✓" if is_correct else "✗"
        
        results_detail.append({
            "index": i + 1,
            "task_id": task_id,
            "is_correct": is_correct,
            "status": process_status,
            "predicted": predicted_grid,
            "ground_truth": gt_grid,
        })
        
        print(f"结果: {'成功' if predicted_grid else '失败'} | 状态: {process_status} | 匹配: {status_icon}")
        time.sleep(1)

    # 3. 统计报告
    accuracy = check_accuracy(all_predictions, all_ground_truths)
    print_report(results_detail, accuracy, len(tasks), MODEL_NAME, columns=["status"])

if __name__ == "__main__":
    target_tasks = None
    main(data_range=target_tasks, method="direct")
# 上面的函数只是作为示例框架，你可以任意修改和实现其中的逻辑