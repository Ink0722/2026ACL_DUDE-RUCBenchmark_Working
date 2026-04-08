import os
import json
import csv

results_dir = './results'
csv_dir = './csv_results'  # 新建保存csv的文件夹

os.makedirs(csv_dir, exist_ok=True)  # 创建文件夹

FILENAME = 'stage1_results_20251230_024904_992303.jsonl'  # 作为常量定义

jsonl_path = os.path.join(results_dir, FILENAME)
# csv_path = os.path.join(csv_dir, FILENAME)
csv_path = os.path.join(csv_dir, os.path.splitext(FILENAME)[0] + '.csv')

with open(jsonl_path, 'r', encoding='utf-8') as fin, open(csv_path, 'w', newline='', encoding='utf-8') as fout:
    writer = None
    for line in fin:
        data = json.loads(line)
        # 跳过 summary 行
        if data.get('__summary__'):
            continue
        if writer is None:
            # 写表头
            writer = csv.DictWriter(fout, fieldnames=data.keys())
            writer.writeheader()
        writer.writerow(data)
print(f'已保存: {csv_path}')