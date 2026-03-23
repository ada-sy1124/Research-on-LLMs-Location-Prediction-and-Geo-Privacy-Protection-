# from huggingface_hub import HfApi
# import os

# # 确认环境变量是否已设置（可选检查）
# print("Using token:", "已找到 ✅" if os.getenv("HUGGINGFACE_HUB_TOKEN") else "未找到 ❌")

# api = HfApi()  # 会自动使用系统中的 HUGGINGFACE_HUB_TOKEN
# info = api.dataset_info("osv5m/osv5m")  # 数据集名称

# # 计算数据集总大小
# total_bytes = sum(file.size for file in info.siblings if file.size is not None)
# print(f"📦 数据集总大小：{total_bytes / (1024**3):.2f} GB")


# from datasets import load_from_disk

# dataset = load_from_disk("/Applications/Documents/geoai/random_streetview_gemini/YES")

# print("样本数量:", len(dataset))


from datasets import load_from_disk

YES_DIR = "/Applications/Documents/geoai/random_streetview_gemini/YES"
SAMPLE_INDEX = 0

dataset = load_from_disk(YES_DIR)
sample = dataset[SAMPLE_INDEX]

print("样本索引:", SAMPLE_INDEX)
print("所有列名:", dataset.column_names)

for column in dataset.column_names:
    if column == "image":
        print(f"{column}: <image>")
    else:
        print(f"{column}: {sample[column]}")

if "image" in sample and sample["image"] is not None:
    sample["image"].show()
