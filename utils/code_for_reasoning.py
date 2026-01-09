import numpy as np
import re

def execute_and_test(code_str, train_examples, test_input):
    """
    执行代码并返回：(是否通过所有训练集, 测试集结果或错误信息)
    """
    # 构造执行命名空间，注入常用库
    namespace = {"np": np, "numpy": np}
    
    try:
        # 执行代码块
        exec(code_str, namespace)
        
        # 检查是否定义了核心函数
        if 'transform' not in namespace:
            return False, "Error: Function 'transform(input_grid)' not defined."
        
        transform = namespace['transform']
        
        # 验证所有训练示例
        for i, ex in enumerate(train_examples):
            user_out = transform(ex['input'])
            if str(user_out) != str(ex['output']):
                return False, f"Validation Failed on Example {i+1}. Input leads to wrong output."
        
        # 验证全部通过，执行测试输入
        test_out = transform(test_input)
        return True, test_out

    except Exception as e:
        return False, f"Runtime Error: {str(e)}"

def extract_code(text):
    """从模型输出中提取 <code> 或 ```python 标签内的代码"""
    code_match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if not code_match:
        code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
    
    if code_match:
        return code_match.group(1).strip()
    return ""