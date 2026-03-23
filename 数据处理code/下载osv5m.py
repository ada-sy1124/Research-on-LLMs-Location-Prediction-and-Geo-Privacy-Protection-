
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


