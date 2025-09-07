import sys

merges_file = "./bpe_tokenizer/tinystories_merges.txt"
print(f"--- 开始诊断文件: {merges_file} ---")

found_problem = False
try:
    with open(merges_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # 跳过空行
            if not line.strip():
                continue

            parts = line.strip().split('\t')

            # 检查分割后的部分数量是否不为 2
            if len(parts) != 2:
                print(f"\n[!!! 问题行找到了 !!!]")
                print(f"  - 行号: {i + 1}")
                # 使用 repr() 可以显示出所有特殊字符，比如换行符、制表符等
                print(f"  - 原始行内容: {repr(line)}")
                print(f"  - strip() 后的内容: {repr(line.strip())}")
                print(f"  - split('\\t') 后的结果: {parts}")
                print(f"  - 期望得到 2 个部分，实际得到 {len(parts)} 个")
                found_problem = True
                # 找到第一个问题后就退出，保持输出干净
                break 
except FileNotFoundError:
    print(f"错误: 文件未找到 at '{merges_file}'")
    print("请先确保你已经成功运行了 run_bpe_training.py")
    sys.exit(1)


if not found_problem:
    print("\n--- 诊断完成 ---")
    print("所有行的格式都正确（都能被 '\\t' 分割成两部分）。如果错误依旧，问题可能在别处。")
else:
    print("\n--- 诊断完成 ---")