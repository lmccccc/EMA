from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gaurav8297/navix",
    repo_type="dataset",
    local_dir="/mnt/data/mocheng/dataset/navix_dataset",
    resume_download=True
)

print("Download completed.")