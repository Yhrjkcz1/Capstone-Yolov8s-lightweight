## 清洗 VisDrone 数据集中的标签文件，删除超出类别范围的标签
import os

root = r"D:/Capstone/Datasets/visdrone"
splits = ["train", "val", "test-dev"]

valid_classes = set(range(10))  # VisDrone DET 的类别：0~9

count_files = 0
count_removed = 0

for split in splits:
    label_dir = os.path.join(root, "labels", split)

    if not os.path.exists(label_dir):
        print(f"跳过 {split}，因为找不到 {label_dir}")
        continue

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, file)
        count_files += 1

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id in valid_classes:
                new_lines.append(line)
            else:
                count_removed += 1

        with open(label_path, "w") as f:
            f.writelines(new_lines)

print(f"已处理标签文件数量: {count_files}")
print(f"已删除超范围类别标签数量: {count_removed}")
print("清洗完成 ✔")
