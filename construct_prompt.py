import json
import re


def construct_critic_prompt(train_data_str, test_input, first_attempt):
    """强化审判官：增加验证示例的要求"""
    return [
        {"role": "system", "content": "You are a senior logic auditor. You excel at finding hidden contradictions in visual reasoning patterns."},
        {"role": "user", "content": f"### TASK CONTEXT\n{train_data_str}\n\n"
                                    f"### TEST INPUT\n{test_input}\n\n"
                                    f"### SOLVER'S PROPOSED SOLUTION\n{first_attempt}\n\n"
                                    f"--- CRITIC INSTRUCTION ---\n"
                                    f"Please audit the solution with high skepticism:\n"
                                    f"1. **Back-test Examples**: Try to apply the solver's rule to Example 1. Does it produce the correct output? If not, the rule is objectively wrong.\n"
                                    f"2. **Grid Geometry Check**: Does the output grid match the necessary dimensions? (e.g., if the pattern is a 3x3 expansion, is the result 3x3?)\n"
                                    f"3. **Color Logic**: Are there any colors in the output that have no logical reason to be there?\n\n"
                                    f"If the solution is 100% flawless across ALL examples, reply ONLY with 'TOTAL_CORRECT'.\n"
                                    f"Otherwise, provide a concise, 'hard-hitting' list of errors."}
    ]

def construct_refine_prompt(original_messages, first_attempt, critic_feedback):
    """强化修正：加入重组逻辑和对比指令"""
    return original_messages + [
        {"role": "assistant", "content": first_attempt},
        {"role": "user", "content": f"### CRITIC'S AUDIT FEEDBACK\n{critic_feedback}\n\n"
                                    f"--- REFINEMENT INSTRUCTION ---\n"
                                    f"The critic has identified fatal flaws in your reasoning. Your previous answer is likely WRONG.\n"
                                    f"1. **Fresh Start**: Completely discard your previous hypothesis. Re-examine the training samples from a new perspective.\n"
                                    f"2. **Verification**: Before generating the final grid, mentally verify your new rule against Example 1.\n"
                                    f"3. **Explanation**: Briefly explain why your previous answer was wrong and what you changed.\n"
                                    f"4. **Final Result**: Place your new corrected 2D array inside <result> tags."}
    ]


def construct_python_code_prompt(d):
    train_data_str = ""
    for i, example in enumerate(d['train']):
        train_data_str += f"# Example {i+1}\ninput_{i+1} = {example['input']}\noutput_{i+1} = {example['output']}\n\n"

    test_input = d['test'][0]['input']
    
    messages = [
        {"role": "system", "content": "You are a Python expert specializing in NumPy and grid manipulation. Your task is to write a transformation function for ARC puzzles."},
        {"role": "user", "content": (
            "Based on the provided input-output examples, identify the geometric or logical rule and implement it as a Python function.\n\n"
            f"{train_data_str}"
            f"test_input = {test_input}\n\n"
            "### REQUIREMENTS\n"
            "1. Define a function `transform(input_grid)` that returns the transformed grid as a nested list.\n"
            "2. Use logic like object detection, coordinate shifting, or color filling.\n"
            "3. Ensure the function works for ALL examples provided.\n"
            "4. Put your code inside <code> tags.\n\n"
            "### ANALYSIS\n"
            "Briefly describe the objects and the transformation rule first, then write the code."
        )}
    ]
    return messages

def construct_prompt(d):
    """
    构造用于大语言模型的提示词
    
    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。
    
    返回:
    list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    示例: [{"role": "system", "content": "系统提示内容"}, 
           {"role": "user", "content": "用户提示内容"}]
    """
    
    train_data_str = ""
    for i, example in enumerate(d['train']):
        train_data_str += f"训练样本 {i+1} 输入: {example['input']}\n"
        train_data_str += f"训练样本 {i+1} 输出: {example['output']}\n"

    test_input = d['test'][0]['input']
    
    # # Fewshot
    # # V1: Low Accuracy Version (Weak Instruction)
    # messages = [
    #     {"role": "system", "content": "You are an AI"},
    #     {"role": "user", "content": f"{train_data_str}\nTest Input: {test_input}\n. Based on the rules above, give the input and return a result."}
    # ]

    # # V2: Higher Accuracy Version (Structured Instruction)
    # messages = [
    #     {"role": "system", "content": "You are an AI"},
    #     {"role": "user", "content": f"Please provide the predicted output for the test input based on the following rules. Return ONLY the grid array itself.\n\nRules:\n{train_data_str}\nTest Input: {test_input}\nResult:"}
    # ]

    # CoT
    messages =  [
        {"role": "system", "content": (
            "You are a world-class solver of the Abstraction and Reasoning Corpus (ARC). "
            "Your intelligence is characterized by strong spatial reasoning and the ability "
            "to identify abstract rules from minimal examples.\n\n"
            "Key Principles for ARC:\n"
            "1. Objects: Identify groups of connected pixels as distinct objects.\n"
            "2. Properties: Note object shapes, colors, and sizes.\n"
            "3. Transformations: Look for movement (translation), rotation, mirroring, "
            "color changes, or pattern repetition.\n"
            "4. Grid Geometry: Pay attention to the grid dimensions and how they might change."
        )},
        {"role": "user", "content": (
            f"### TRAINING EXAMPLES\n{train_data_str}\n"
            f"### TEST INPUT GRID\n{test_input}\n\n"
            "### STEP-BY-STEP REASONING\n"
            "1. Analyze each training example: What is the transformation? (Describe in terms of objects and grid changes).\n"
            "2. Verify the rule: Does your rule work perfectly for ALL training examples?\n"
            "3. Apply to test input: Predict the output grid based on your verified rule.\n\n"
            "Provide the final 2D array inside <result> tags. Start your analysis now:"
        )}
    ]
    
    return messages



def extract_grid_from_string(s):
    """辅助函数：从字符串中提取并验证 Python 列表格式"""
    try:
        # 寻找 [[...]] 结构
        match = re.search(r'(\[\[\s*.*?\s*\]\])', s, re.DOTALL)
        if match:
            grid_str = match.group(1)
            # 处理可能的尾部逗号等格式不规范问题
            grid_str = grid_str.replace('\n', '').strip()
            return json.loads(grid_str)
    except Exception:
        return None
    return None

def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本
    
    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    示例: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    """
    #print(f"\n--- 模型输出全文 ---\n{text}\n---")

    # 策略 1：优先匹配 <result> 标签内的内容
    tag_match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if tag_match:
        content = tag_match.group(1).strip()
        # 尝试从标签内容中提取数组
        grid = extract_grid_from_string(content)
        if grid:
            print("成功从 <result> 标签中解析网格")
            return grid

    # 策略 2：如果没标签或解析失败，尝试全文搜索第一个符合二维数组格式的内容
    # 注意：这里可能会误抓“步骤 1”中的内容
    print("未发现有效标签内容，尝试全文正则搜索...")
    grid = extract_grid_from_string(text)
    if grid:
        return grid

    print("解析失败：未能找到有效网格")
    return []