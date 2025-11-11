# encoding=utf-8
import shutil

import cv2
import yaml
import os
from pathlib import Path
from collections import Counter
import albumentations as A

def count_classes(label_dir: Path):
    """统计当前数据集中每个类别的标注框数量"""
    counter = Counter()
    for txt in label_dir.glob("*.txt"):
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                class_id = int(float(parts[0]))
                counter[class_id] += 1
    return counter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # global
    kagglehub_crop_pests_dataset_path = r'C:\Users\shenyanjian\.cache\kagglehub\datasets\rupankarmajumdar\crop-pests-dataset\versions\2'
    train_path = os.path.join(kagglehub_crop_pests_dataset_path, 'train')
    train_images_path = os.path.join(train_path, 'images')
    train_labels_path = os.path.join(train_path, 'labels')

    # 任务一：删除我们周一上午的删除的打标框
    with open('data.yaml', 'r', encoding='utf-8') as f:
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
    print("各类别标注框数量：")
    for cls_id, count in sorted(class_counter.items()):
        print(f"类别 {cls_id}: {count} 个标注框")

    # 可选：统计总数
    print("\n总标注框数:", sum(class_counter.values()))
    print("总类别数:", len(class_counter))

    # 任务三：对少数类进行增强
    with open(os.path.join(kagglehub_crop_pests_dataset_path, 'data.yaml'), 'r', encoding='utf-8') as f:
        # 这个data.yaml是数据集的那个类别yaml
        data_yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    print(data_yaml_dict)
    class_names = data_yaml_dict['names']
    max_class_count = max(class_counter.values())
    # 定义增强策略
    augment = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.MotionBlur(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    round_num = 0
    while True:
        round_num += 1
        class_counter = count_classes(train_labels_path)
        max_class = max(class_counter.values())
        min_class = min(class_counter.values())
        ratio = max_class / min_class
        print(f"\n=== 第 {round_num} 轮增强 ===")
        print("当前类别统计：")
        for cid, cnt in sorted(class_counter.items()):
            print(f"  {class_names[cid]:10s}: {cnt}")
        print(f"最大类: {max_class}, 最小类: {min_class}, 比例差距: {ratio:.2f}")

        # 判断是否已达平衡阈值
        if ratio <= 1.4:
            print("\n✅ 达到平衡目标，增强结束。")
            break

        # 逐类增强：少于 max_class / 1.4 的类需要增强
        for cls_id, count in class_counter.items():
            cls_name = class_names[cls_id]
            if count * 1.4 >= max_class:
                continue  # 已接近，不增强
            target_add = int(count * 0.1)  # 每轮增加 10%
            print(f"\n增强类别 {cls_name}：当前 {count}，计划新增约 {target_add}")

            # 找出含该类的图片
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

                # 随机增强多次，直到达到目标数或略超
                for i in range(target_add // max(1, len(bboxes)) + 1):
                    aug_result = augment(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = aug_result["image"]
                    aug_bboxes = aug_result["bboxes"]
                    aug_labels = aug_result["class_labels"]

                    new_name = f"{img_path.stem}_aug{round_num}_{i}"
                    new_img_path = Path(train_images_path) / f"{new_name}.jpg"
                    new_label_path = Path(train_labels_path) / f"{new_name}.txt"

                    cv2.imwrite(str(new_img_path), aug_img)
                    with open(new_label_path, "w") as f:
                        for cid, bbox in zip(aug_labels, aug_bboxes):
                            f.write(f"{int(cid)} {' '.join(map(lambda x: f'{x:.6f}', bbox))}\n")

        print(f"\n--- 第 {round_num} 轮增强完成，重新统计类别 ---")

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
    print("各类别标注框数量：")
    for cls_id, count in sorted(class_counter.items()):
        print(f"类别 {cls_id}: {count} 个标注框")

    # 可选：统计总数
    print("\n总标注框数:", sum(class_counter.values()))
    print("总类别数:", len(class_counter))
