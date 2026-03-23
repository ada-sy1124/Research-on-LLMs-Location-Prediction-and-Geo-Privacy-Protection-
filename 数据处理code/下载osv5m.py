# from huggingface_hub import snapshot_download
# import os

# # === 你要存放数据集的位置（请改这里）===
# save_dir = "/Users/sy1124/Downloads/Documents/geoai/osv5m"   # ←⚠️改为你的磁盘路径，例如 "/Volumes/DATA/osv5m"

# # === 如果目录不存在则创建 ===
# os.makedirs(save_dir, exist_ok=True)

# print(f"📁 数据将下载到: {save_dir}")
# print("⚠️ 请确认你的硬盘至少有 1.5 TB 可用空间。\n")

# # === 开始完整下载 ===
# snapshot_download(
#     repo_id="osv5m/osv5m",   # 数据集名称
#     repo_type="dataset",     # 数据集类型
#     local_dir=save_dir,      # 保存位置
#     resume_download=True,    # 支持断点续传（建议强烈开）
#     max_workers=16           # 多线程并行下载（可根据 CPU 调大/调小）
# )

# print("\n✅ 下载完成！")
# print(f"数据已保存在: {save_dir}")
# print("你可以使用 streaming 模式直接加载，而不占用内存。")


from huggingface_hub import hf_hub_download

# 下载第一个训练分片 00.zip 到本地文件夹 ./osv5m_first_zip
zip_path = hf_hub_download(
    repo_id="osv5m/osv5m",
    filename="00.zip",
    subfolder="images/train",
    repo_type="dataset",
    local_dir="/Users/sy1124/Downloads/Documents/geoai/osv5m "
)

print("✅ 已下载:", zip_path)


