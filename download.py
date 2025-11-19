
import kagglehub

# The main task is to download this dataset. The default download location for the data is:
# /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/
path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
print(path)


