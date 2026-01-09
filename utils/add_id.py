import json
import os

def add_ids_to_jsonl(input_path, output_path, start_id=0):
    """
    读取 jsonl 文件，为每一行添加递增的 id 属性并保存。
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return

    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            
            # 解析 JSON
            data = json.loads(line)
            
            # 注入 id 属性
            # 如果原数据已有 id 且你不想覆盖，可以加个判断：if 'id' not in data:
            data['id'] = start_id + i
            
            # 写入新文件
            json_line = json.dumps(data, ensure_ascii=False)
            outfile.write(json_line + '\n')
            processed_count += 1

    print(f"处理完成！")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"共为 {processed_count} 条数据添加了 ID。")

if __name__ == "__main__":
    # --- 配置区域 ---
    INPUT_FILE = "val.jsonl"       # 你的原始文件名
    OUTPUT_FILE = "val_with_id.jsonl" # 处理后的文件名
    
    add_ids_to_jsonl(INPUT_FILE, OUTPUT_FILE)