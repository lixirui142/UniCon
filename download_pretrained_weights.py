import os
from huggingface_hub import snapshot_download

weight_dir = "./weights"
os.makedirs(weight_dir, exist_ok=True)
snapshot_download(repo_id="lixirui142/unicon", local_dir=weight_dir)
