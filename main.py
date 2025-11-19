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

    # Task 1: Delete the deletion markers from our Monday morning assignments.
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
            # print("ERROR", single_mistake_pic, jpg_path, img_flag, txt_path, label_flag)
            pass
        else:
            os.remove(jpg_path)
            print("Removed", jpg_path)
            os.remove(txt_path)
            print("Removed", txt_path)

    # Task 2: Count the number of boxes in each category.
    class_counter = Counter() # Initialize the counter

    for txt_file in Path(train_labels_path).rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])  # The first number is the category number
                class_counter[class_id] += 1
    # Print results
    print("Number of label boxes for each category:")
    for cls_id, count in sorted(class_counter.items()):
        print(f"Category {cls_id}: {count} label boxes")

    # Optional: Count the total
    print("\nTotal number of annotation boxes:", sum(class_counter.values()))
    print("Total number of categories:", len(class_counter))

    # Task 3: Dynamically enhance the minority classes until the difference between the number of elements in the smallest and largest classes is <= 1.4 times.
    with open(os.path.join(kagglehub_crop_pests_dataset_path, 'data.yaml'), 'r', encoding='utf-8') as f:
        data_yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    class_names = data_yaml_dict['names']

    augment = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.MotionBlur(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3,
                                clip=True
    ))

    round_idx = 0
    while True:
        # Recalculation
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

        # Exit when equilibrium condition is reached
        if ratio <= 1.4:
            print("\n All categories are now basically balanced; enhancements are complete.")
            break

        # Enhance minority classes by 10%
        for cls_id, count in class_counter.items():
            if count * 1.4 >= max_class_count:
                continue  # Nearing equilibrium
            increase_num = max(1, int(count * 0.1))  # Increase by 10% each time
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
                    break  # Current class enhancement is at least 10%

        round_idx += 1

    # Task 4: Recount the number of boxes in each category.
    class_counter = Counter() # Initialize the counter

    for txt_file in Path(train_labels_path).rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])  # The first number is the category number
                class_counter[class_id] += 1
    # Print results
    print("Number of label boxes for each category:")
    for cls_id, count in sorted(class_counter.items()):
        print(f"Category {cls_id}: {count} label box")

    # Optional: Count the total
    print("\nTotal number of annotation boxes:", sum(class_counter.values()))
    print("Total number of categories:", len(class_counter))
