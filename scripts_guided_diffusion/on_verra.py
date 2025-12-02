import kagglehub

# Download latest version
path = kagglehub.dataset_download("aiocta/brats2023-part-1")

print("Path to dataset files:", path)
