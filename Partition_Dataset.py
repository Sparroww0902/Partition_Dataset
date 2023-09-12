import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# 创建文件夹
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)

# 设置数据集路径和目标文件夹路径
dataset_path = "D:/PyCharm/Py_Projects/Partition_Dataset"
train_images_path = "images/train"
val_images_path = "images/val"
train_labels_path = "labels/train"
val_labels_path = "labels/val"

# 获取图像和标签文件列表
image_files = [f for f in os.listdir(os.path.join(dataset_path, "JPEGImages")) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(os.path.join(dataset_path, "Annotations")) if f.endswith(".xml")]
random.shuffle(image_files)

# 计算验证集和训练集的分割点
split_ratio = 0.7
split_index = int(len(image_files) * split_ratio)

# 划分图像文件列表为训练集和验证集
train_image_files = image_files[:split_index]
val_image_files = image_files[split_index:]

# 转换VOC XML标签为YOLOv5格式的TXT标签
def convert_voc_to_yolov5(xml_path, image_width, image_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.findall("object"):
        class_id = obj.find("name").text
        if class_id == 'water leak':
            class_id = 0
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / (2 * image_width)
        y_center = (ymin + ymax) / (2 * image_height)
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return labels

# 复制训练集图像和转换后的标签
for image_file in train_image_files:
    image_path = os.path.join(dataset_path, "JPEGImages", image_file)
    xml_path = os.path.join(dataset_path, "Annotations", image_file.replace(".jpg", ".xml"))
    image = Image.open(image_path)
    image_width, image_height = image.size

    yolo_labels = convert_voc_to_yolov5(xml_path, image_width, image_height)

    with open(os.path.join(train_labels_path, image_file.replace(".jpg", ".txt")), "w") as f:
        f.write("\n".join(yolo_labels))

    shutil.copy(image_path, train_images_path)

# 复制验证集图像和转换后的标签
for image_file in val_image_files:
    image_path = os.path.join(dataset_path, "JPEGImages", image_file)
    xml_path = os.path.join(dataset_path, "Annotations", image_file.replace(".jpg", ".xml"))
    image = Image.open(image_path)
    image_width, image_height = image.size

    yolo_labels = convert_voc_to_yolov5(xml_path, image_width, image_height)

    with open(os.path.join(val_labels_path, image_file.replace(".jpg", ".txt")), "w") as f:
        f.write("\n".join(yolo_labels))

    shutil.copy(image_path, val_images_path)

print("数据集划分和标签转换完成！")
