import kagglehub

# Download latest version
path = kagglehub.dataset_download("emirhanai/2024-u-s-election-sentiment-on-x")

print("Path to dataset files:", path)