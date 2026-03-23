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