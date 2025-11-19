
This script is based on data cleaning we collaborated on on November 10, 2025, and performs **data preprocessing steps** on the "rupankarmajumdar/crop-pests-dataset" dataset.

# Environmental Preparation

    pip install albumentations kagglehub

# How to download data

    pip install -U pip setuptools wheel

```
import kagglehub

path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
# Print the downloaded path
print("Dataset downloaded to:", path)
```

Default download location for Linux datasets:

    /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/

Windows dataset default download location:

    C:\Users\{USER_NAME}\.cache\kagglehub\datasets\rupankarmajumdar\crop-pests-dataset\versions\2\

If you process the data yourself later and it becomes corrupted, please download the data again.

Delete `# /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/`. Make sure to delete the `/2/` directory completely. Then run `download.py`. This will allow you to retrieve the new data.

# Category Balance

Liu has uploaded the image set to YOLO Hub: `https://hub.ultralytics.com/datasets/dL8pDaojLoJ4WFIuNV2l`, which can be viewed for labeling reference.

As shown in the figure, the following quantities represent the number of bounding boxes (not the number of images), indicating an imbalance.

![img.png](imgs/img.png)

After categorization and additional enhancement for minority classes, there will be image names containing `aug0` such as `Weevil-76-_jpg.rf.b62dddb58ffca392445b8c39db8b9788_aug0.jpg`.

![img_1.png](imgs/img_1.png)

![img_3.png](imgs/img_3.png)

I personally checked the completed script and confirmed that it has passed the YOLO Hub annotation check. I've confirmed that the coordinate bounding boxes for these enhanced files (files with the .aug extension) have been correctly transformed, so there's no need to worry about incorrect annotations in my program.

As shown in the image, I specifically compiled the .aug images for checking. It's evident that the bounding box coordinates have been correctly transformed along with the original image.

![](./imgs/32307710772100.png)


# Enhancement

The enhancement principle uses **albumentations**, which can be found in the team's documentation. It involves processing images and labels through rotation, changes in lighting, and other methods.

![img_8.png](imgs/img_8.png)

Because the dynamic algorithm enhances a minority of categories with a 20% step size each round, and the difference between the minimum and maximum number of categories is controlled to be no more than 1.4, it is considered balanced.

Because there is a possibility of "aug after aug", image paths (including img and label) may contain multiple `aug` suffixes as shown below.

![img_7.png](imgs/img_7.png)

# Data Cleaning

Configuration options are for reference; the main thing is that you need to change the address of your own dataset.

1. The main function in main.py

# How to run

python main.py

After the script finishes running, the categories will be basically balanced.

![img_6.png](imgs/img_6.png)





