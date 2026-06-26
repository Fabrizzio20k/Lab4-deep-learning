from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-8B"
local_dir = "./qwen_model_cache"

snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
