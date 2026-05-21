import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from huggingface_hub import snapshot_download

# 下载整个模型仓库
model_path = snapshot_download(
    repo_id="TimoZhou1024/werewolf-kto-lora2",
    local_dir="./models/werewolf-kto-lora2",  # 本地保存路径
    local_dir_use_symlinks=False,      # 避免使用符号链接，直接复制文件
    resume_download=True,              # 支持断点续传
)

print(f"模型已下载到: {model_path}")