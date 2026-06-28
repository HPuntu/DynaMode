from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="compsciencelab/mdCATH",
    repo_type="dataset",
    local_dir="/projects/u6hf/hew/mdCATH/data",
)