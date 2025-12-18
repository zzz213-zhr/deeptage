# import onnx
# m = onnx.load("deeptage.onnx")
# print(m.graph.input)

# from torchvision.datasets import CIFAR10
# import torchvision.transforms as T
# from PIL import Image
#
# ds = CIFAR10(root='.', train=False, download=True)
# img, label = ds[0]
# img.save("test_cat.png")

import pickle
import os
import numpy as np
from PIL import Image

# 路径设置
CIFAR_DIR = "data/cifar-10-batches-py"
OUT_DIR = "data/cifar10_demo_images"
SPLIT = "test_batch"            # 使用测试集

os.makedirs(OUT_DIR, exist_ok=True)

# CIFAR-10 类别
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NUM_PER_CLASS = 5

# 初始化计数器
class_count = {name: 0 for name in label_names}

# 读取 CIFAR batch
with open(os.path.join(CIFAR_DIR, SPLIT), "rb") as f:
    batch = pickle.load(f, encoding="bytes")

data = batch[b"data"]
labels = batch[b"labels"]

# 导出图片
for i in range(len(data)):
    label = labels[i]
    class_name = label_names[label]
    if class_count[class_name] >= NUM_PER_CLASS:
        continue
    img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
    img_pil = Image.fromarray(img)
    class_count[class_name] += 1
    save_name = f"{class_name}_{class_count[class_name]}.png"
    save_path = os.path.join(OUT_DIR, save_name)
    img_pil.save(save_path)
    if all(v >= NUM_PER_CLASS for v in class_count.values()):
        break

print("CIFAR-10 演示图片导出完成")
print("导出目录：", OUT_DIR)
