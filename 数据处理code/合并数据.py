# from datasets import load_from_disk, concatenate_datasets
# import os

# # === 输入输出路径 ===
# chunks_dir = "/Users/sy1124/Downloads/Documents/geoai/first_filter_test"  # chunk 文件夹所在目录
# output_path = "/Users/sy1124/Downloads/Documents/geoai/合并后街景数据集"  # 合并后输出目录

# # === 收集所有 chunk 路径 ===
# chunk_paths = [
#     os.path.join(chunks_dir, d)
#     for d in os.listdir(chunks_dir)
#     if d.startswith("chunk_")
# ]

# chunk_paths.sort()  # 按数字顺序排序（可选）

# print(f"发现 {len(chunk_paths)} 个 chunk:")
# for p in chunk_paths:
#     print(" -", p)

# # === 逐个 load 并 concat ===
# datasets = [load_from_disk(p) for p in chunk_paths]
# merged = concatenate_datasets(datasets)

# # === 保存合并后的数据集 ===
# merged.save_to_disk(output_path)

# print("✅ 合并完成！已保存至：", output_path)
# print("📦 合并后样本数量：", len(merged))















# import os
# from datasets import load_from_disk, concatenate_datasets

# # =========================
# # 1) 路径配置
# # =========================
# # 稍后处理 NO 数据时，只需更改这两行
# INPUT_DIR = "/Applications/Documents/random_streetview_gemini/YES"
# OUTPUT_DIR = "/Applications/Documents/random_streetview_gemini/YES_merged"

# def main():
    

#     # 获取所有以 "chunk_" 开头的文件夹
#     chunk_folders = [f for f in os.listdir(INPUT_DIR) if f.startswith("chunk_")]
 

#     # ★关键：必须按 chunk 后面的数字进行正确排序
#     # 否则系统的字符排序会把 chunk_10 排在 chunk_2 前面，导致数据顺序错乱
#     chunk_folders.sort(key=lambda x: int(x.split("_")[1]))

#     datasets_list = []
#     print(f"\n🚀 准备合并 {INPUT_DIR} 中的 {len(chunk_folders)} 个 chunk...")
    
#     for chunk_name in chunk_folders:
#         chunk_path = os.path.join(INPUT_DIR, chunk_name)
#         try:
#             ds = load_from_disk(chunk_path)
#             datasets_list.append(ds)
#             print(f"  ✅ 已加载 {chunk_name} (数据量: {len(ds)})")
#         except Exception as e:
#             print(f"  ❌ 加载 {chunk_name} 失败: {e}")

#     # =========================
#     # 2) 拼接与保存
#     # =========================
#     if datasets_list:
#         print("\n🔄 正在拼接数据集...")
#         merged_dataset = concatenate_datasets(datasets_list)
        
#         print(f"💾 正在保存合并后的数据到: {OUTPUT_DIR}")
#         merged_dataset.save_to_disk(OUTPUT_DIR)
        
#         print(f"🎉 合并完成！合并后总数据量: {len(merged_dataset)}")
    

# if __name__ == "__main__":
#     main()





import os
from datasets import load_from_disk, concatenate_datasets

# 指向包含所有 chunk 的父文件夹
INPUT_DIR = "/Applications/Documents/random_streetview_gemini/NO"
# 合并后的新数据保存位置
OUTPUT_DIR = "/Applications/Documents/random_streetview_gemini/NO_merged"

def main():
    # 1. 获取 YES 目录下所有的 chunk 文件夹名称
    chunk_folders = [f for f in os.listdir(INPUT_DIR) if f.startswith("chunk_")]
    
    # 2. 按照数字顺序排序 (确保 chunk_0 在 chunk_1 前面，而不是 chunk_10)
    chunk_folders.sort(key=lambda x: int(x.split("_")[1]))

    datasets_list = []
    
    # 3. 遍历读取每个 chunk 文件夹
    for chunk_name in chunk_folders:
        chunk_path = os.path.join(INPUT_DIR, chunk_name) # 例如: .../YES/chunk_0
        print(f"正在加载: {chunk_path}")
        
        # load_from_disk 会自动处理里面的 .arrow 和 .json 文件
        ds = load_from_disk(chunk_path)
        datasets_list.append(ds)

    # 4. 拼接并保存
    if datasets_list:
        print("\n🔄 正在拼接所有数据...")
        merged_dataset = concatenate_datasets(datasets_list)
        
        print(f"💾 正在保存到: {OUTPUT_DIR}")
        merged_dataset.save_to_disk(OUTPUT_DIR)
        
        print(f"🎉 成功！合并后的总数据量: {len(merged_dataset)} 条")

if __name__ == "__main__":
    main()