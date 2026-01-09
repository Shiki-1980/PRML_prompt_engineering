
import ast
import concurrent.futures
from collections import Counter
from utils.code_for_reasoning import *
from construct_prompt import *
from main import speak_and_listen


def get_self_consistency_result(messages, model_name, sample_count=5):
    """
    通过并行采样和多数投票实现自我一致性 (Self-Consistency)。
    
    返回:
        final_grid (list): 票数最高的预测网格。
        status (str): 投票状态描述。
    """
    samples = []
    
    # 1. 并行化采样
    # 使用 ThreadPoolExecutor 同时发出多个 API 请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=sample_count) as executor:
        # 提交并发任务
        futures = [executor.submit(speak_and_listen, messages, model_name) for _ in range(sample_count)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                reply_text = future.result()
                if reply_text:
                    grid = parse_output(reply_text)
                    if grid:  # 只有成功解析为二维数组的才计入选票
                        # 将列表转为字符串以便 Counter 进行哈希统计
                        samples.append(str(grid))
            except Exception as e:
                print(f"\n采样线程执行出错: {e}")

    # 2. 投票逻辑
    final_grid = []
    status = "解析全失败"

    if not samples:
        return final_grid, status

    # 统计每个候选答案出现的频率
    counts = Counter(samples)
    # 获取票数最高的答案及票数
    most_common_str, vote_count = counts.most_common(1)[0]
    num_unique_answers = len(counts)
    num_total_samples = len(samples)

    # 3. 判定共识状态
    if vote_count > 1:
        status = f"多数票({vote_count}/{num_total_samples})"
    elif num_unique_answers == num_total_samples and num_total_samples > 1:
        status = "完全分歧"
    else:
        status = "无共识(仅1票)"

    # 将选出的字符串答案还原为 Python 列表
    try:
        final_grid = ast.literal_eval(most_common_str)
        print(f"还原网格数组成功:\n {final_grid}")
    except Exception as e:
        print(f"还原网格数组失败: {e}")
        status = "解析还原失败"

    return final_grid, status

def get_reflexion_result(d, model_name):
    """增加二轮强力反思逻辑，应对“死不认错”的情况"""
    
    messages = construct_prompt(d)
    train_data_str = ""
    for i, ex in enumerate(d['train']):
        train_data_str += f"Example {i+1} Input:{ex['input']} Output:{ex['output']}\n"
    test_input = d['test'][0]['input']

    # --- Step 1: 初稿 ---
    print(" (Thinking...) ", end="", flush=True)
    first_draft = speak_and_listen(messages, model_name)
    first_grid = parse_output(first_draft)
    if first_grid == d['test'][0]['output']:
        print(" [Direct Hit!] ", end="")
    # --- Step 2: 审判 ---
    print(" (Reviewing...) ", end="", flush=True)
    critic_msgs = construct_critic_prompt(train_data_str, test_input, first_draft)
    feedback = speak_and_listen(critic_msgs, model_name)
    
    if "TOTAL_CORRECT" in feedback:
        print(" [Passed] ", end="")
        return first_grid, "Passed Directly"

    # --- Step 3: 第一次修正 ---
    print(" [Refining...] ", end="", flush=True)
    refine_msgs = construct_refine_prompt(messages, first_draft, feedback)
    second_draft = speak_and_listen(refine_msgs, model_name)
    second_grid = parse_output(second_draft)

    # --- Step 4: 逻辑判定与可能的“二次强修” ---
    status = "Refined"
    
    if first_grid == second_grid:
        # 触发硬核修正提示
        print(" [Ineffective, retrying...] ", end="", flush=True)
        stern_msg = refine_msgs + [
            {"role": "assistant", "content": second_draft},
            {"role": "user", "content": "WARNING: Your output did not change at all. The critic's feedback was NOT addressed. "
                                        "You are likely stuck in a logic loop. Change your perspective! "
                                        "Look at the spatial relations again. Provide a NEW and DIFFERENT grid."}
        ]
        final_draft = speak_and_listen(stern_msg, model_name)
        final_grid = parse_output(final_draft)
        
        if final_grid == second_grid:
            status = "Refinement Failed (Stubborn)"
        else:
            status = "Refined (After Stern Warning)"
    else:
        final_grid = second_grid
        print(" [Updated] ", end="")

    return final_grid, status

def get_code_result(d, model_name):
    """
    Code-based Reflexion: 
    Solver(Code) -> Executor(Test) -> Error Feedback -> Refiner(Fix Code)
    """
    # 1. 初始化提示词
    messages = construct_python_code_prompt(d)
    
    # --- 第一轮：写代码 ---
    print(" (Coding...) ", end="", flush=True)
    response_1 = speak_and_listen(messages, model_name)
    code_1 = extract_code(response_1)
    print(f" [Code Generated] ", end="", flush=True)
    # 尝试执行
    success_1, result_1 = execute_and_test(code_1, d['train'], d['test'][0]['input'])
    
    if success_1:
        print(" [Verified!] ", end="")
        return result_1, "Code Verified"

    # --- 第二轮：根据报错修正 (Refinement) ---
    print(f" [Code Error, Debugging...] ", end="", flush=True)
    
    # 构造 Debug 提示词，反馈具体的错误信息
    debug_messages = messages + [
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": (
            f"Your previous code failed. \n"
            f"FEEDBACK: {result_1}\n"
            "Please analyze why the logic failed, reconsider the grid transformation, "
            "and provide the complete corrected Python code in <code> tags."
        )}
    ]
    
    response_2 = speak_and_listen(debug_messages, model_name)
    code_2 = extract_code(response_2)
    
    # 再次尝试执行
    success_2, result_2 = execute_and_test(code_2, d['train'], d['test'][0]['input'])
    
    if success_2:
        print(" [Fixed Successfully!] ", end="")
        return result_2, "Fixed via Debug"
    else:
        # 如果还是错，我们选择解析第二次生成的文本答案作为保底，或者直接返回空
        print(" [Still Failed] ", end="")
        return [], "Code Refinement Failed"

def get_hybrid_sc_reflexion_result(d, model_name, sample_count=3):
    """
    结合 Self-Consistency 和 Reflexion 的混合模式：
    1. 并行采样多个结果并投票选出最优解。
    2. 将最优解送入反思流程进行逻辑校验和修正。
    """
    messages = construct_prompt(d)
    train_data_str = ""
    for i, ex in enumerate(d['train']):
        train_data_str += f"Example {i+1} Input:{ex['input']} Output:{ex['output']}\n"
    test_input = d['test'][0]['input']

    # --- Step 1: 并行采样与投票 (Self-Consistency) ---
    print(f" (Sampling x{sample_count}...) ", end="", flush=True)
    sc_grid, sc_status = get_self_consistency_result(messages, model_name, sample_count=sample_count)
    
    if not sc_grid:
        return [], "SC_Failed_No_Grid"

    # --- Step 2: 审判官介入 (Critic) ---
    print(f" (Reflecting on Majority Vote...) ", end="", flush=True)
    # 这里构造一个模拟的回复文本，让 Critic 以为这是模型的输出
    pseudo_reply = f"Thinking process: Based on majority vote.\nFinal Grid: {sc_grid}"
    
    critic_msgs = construct_critic_prompt(train_data_str, test_input, pseudo_reply)
    feedback = speak_and_listen(critic_msgs, model_name)

    # 如果审判官认为是对的，直接返回投票结果
    if "TOTAL_CORRECT" in feedback:
        print(" [SC Verified] ", end="")
        return sc_grid, f"SC_{sc_status}_Verified"

    # --- Step 3: 基于反馈进行修正 (Refine) ---
    print(" [SC Rejected -> Refining...] ", end="", flush=True)
    refine_msgs = construct_refine_prompt(messages, pseudo_reply, feedback)
    final_reply = speak_and_listen(refine_msgs, model_name)
    final_grid = parse_output(final_reply)

    # 逻辑判定
    if final_grid == sc_grid:
        status = f"SC_{sc_status}_Unchanged_After_Refine"
    else:
        status = f"SC_{sc_status}_Refined_to_New"

    return final_grid, status

def solve_task(method_type, task_data, model_name, **kwargs):
    """
    统一调度器：根据 method_type 选择不同的求解策略
    """
    if method_type == "direct":
        # 基础模式：单次请求
        from construct_prompt import construct_prompt, parse_output
        messages = construct_prompt(task_data)
        reply = speak_and_listen(messages, model_name)
        return parse_output(reply), "Direct_Shot"

    elif method_type == "sc":
        # 自我一致性模式
        from construct_prompt import construct_prompt
        messages = construct_prompt(task_data)
        sample_count = kwargs.get("sample_count", 3)
        return get_self_consistency_result(messages, model_name, sample_count)

    elif method_type == "reflexion":
        # 纯反思模式
        return get_reflexion_result(task_data, model_name)

    elif method_type == "code":
        # 代码执行模式
        return get_code_result(task_data, model_name)

    elif method_type == "hybrid":
        # 混合模式 (SC + Reflexion)
        sample_count = kwargs.get("sample_count", 3)
        return get_hybrid_sc_reflexion_result(task_data, model_name, sample_count)

    else:
        raise ValueError(f"未知的方法类型: {method_type}")
    

