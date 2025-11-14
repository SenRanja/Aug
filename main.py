# encoding=utf-8
import shutil

import cv2
import yaml
import os
from pathlib import Path
from collections import Counter
import albumentations as A
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # global
    kagglehub_crop_pests_dataset_path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\shenyanjian\.cache\kagglehub\datasets\rupankarmajumdar\crop-pests-dataset\versions\2'
    print(kagglehub_crop_pests_dataset_path)


    train_path = os.path.join(kagglehub_crop_pests_dataset_path, 'train')
    train_images_path = os.path.join(train_path, 'images')
    train_labels_path = os.path.join(train_path, 'labels')

    # 任务一：删除我们周一上午的删除的打标框
    with open('./data.yaml', 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        f.close()

    # print(yaml_dict['mistake_pic'])
    for single_mistake_pic in yaml_dict['mistake_pic']:
        jpg_path = os.path.join(train_images_path, f"{Path(single_mistake_pic).stem}.jpg")
        img_flag = os.path.exists(jpg_path)
        txt_path = os.path.join(train_labels_path, f"{Path(single_mistake_pic).stem}.txt")
        label_flag = os.path.exists(txt_path)
        if not (img_flag and label_flag):
            # label和img无法对应上的，此处打印
            # 如果运行过后可以注释
            # print("ERROR", single_mistake_pic, jpg_path, img_flag, txt_path, label_flag)
            pass
        else:
            os.remove(jpg_path)
            print("Removed", jpg_path)
            os.remove(txt_path)
            print("Removed", txt_path)

    # 任务二：统计当前各类别打框数量
    class_counter = Counter() # 初始化计数器

    for txt_file in Path(train_labels_path).rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])  # 第一个数是类别编号
                class_counter[class_id] += 1
    # 打印结果
    print("Number of label boxes for each category:")
    for cls_id, count in sorted(class_counter.items()):
        print(f"Category {cls_id}: {count} label boxes")

    # 可选：统计总数
    print("\nTotal number of annotation boxes:", sum(class_counter.values()))
    print("Total number of categories:", len(class_counter))

    # 任务三：对少数类进行动态增强，直到最小类与最大类数量差距 <= 1.4倍
    with open(os.path.join(kagglehub_crop_pests_dataset_path, 'data.yaml'), 'r', encoding='utf-8') as f:
        data_yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    class_names = data_yaml_dict['names']

    augment = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.MotionBlur(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3,
                                clip=True  # ✅ 新增：自动裁剪越界坐标到 [0,1]
    ))

    round_idx = 0
    while True:
        # 重新统计
        class_counter = Counter()
        for txt_file in Path(train_labels_path).rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    class_id = int(float(line.strip().split()[0]))
                    class_counter[class_id] += 1

        max_class_count = max(class_counter.values())
        min_class_count = min(class_counter.values())
        ratio = max_class_count / min_class_count

        print(f"\n {round_idx + 1} th statistics after round enhancement:")
        for cls_id, count in sorted(class_counter.items()):
            print(f"Category {cls_id} ({class_names[cls_id]}): {count} label boxes")
        print(f"Current largest class: {max_class_count} Minimal class:{min_class_count} Difference ratio:{ratio:.2f}")

        # 达到平衡条件则退出
        if ratio <= 1.4:
            print("\n All categories are now basically balanced; enhancements are complete.")
            break

        # 对少数类增强10%
        for cls_id, count in class_counter.items():
            if count * 1.4 >= max_class_count:
                continue  # 已接近平衡
            increase_num = max(1, int(count * 0.1))  # 每次增加10%
            cls_name = class_names[cls_id]
            print(f"\nEnhanced Category {cls_name} (current {count}, Increase by approximately {increase_num})")

            processed = 0
            for label_path in Path(train_labels_path).glob("*.txt"):
                with open(label_path, "r") as f:
                    lines = [line.strip().split() for line in f if line.strip()]
                if not any(int(float(line[0])) == cls_id for line in lines):
                    continue

                img_path = Path(train_images_path) / f"{label_path.stem}.jpg"
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                bboxes, class_labels = [], []
                for parts in lines:
                    class_labels.append(int(float(parts[0])))
                    bboxes.append(list(map(float, parts[1:])))

                aug_result = augment(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = aug_result["image"]
                aug_bboxes = aug_result["bboxes"]
                aug_labels = aug_result["class_labels"]

                new_name = f"{img_path.stem}_aug{round_idx}_{processed}"
                new_img_path = Path(train_images_path) / f"{new_name}.jpg"
                new_label_path = Path(train_labels_path) / f"{new_name}.txt"

                cv2.imwrite(str(new_img_path), aug_img)
                with open(new_label_path, "w") as f:
                    for cid, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{int(cid)} {' '.join(map(lambda x: f'{x:.6f}', bbox))}\n")

                processed += 1
                if processed >= increase_num:
                    break  # 当前类增强够10%

        round_idx += 1

    # 任务四：再次统计当前各类别打框数量
    class_counter = Counter() # 初始化计数器

    for txt_file in Path(train_labels_path).rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])  # 第一个数是类别编号
                class_counter[class_id] += 1
    # 打印结果
    print("Number of label boxes for each category:")
    for cls_id, count in sorted(class_counter.items()):
        print(f"Category {cls_id}: {count} label box")

    # 可选：统计总数
    print("\nTotal number of annotation boxes:", sum(class_counter.values()))
    print("Total number of categories:", len(class_counter))
