
import kagglehub

# 主要就是下载这个数据集，数据默认下载位置：
# /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/
path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
print("Dataset downloaded to:", path)


