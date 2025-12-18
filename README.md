## 项目概述
```
该项目是一个基于 DeepTage 论文和 CIFAR-10 数据集的简要实现项目，支持模型训练、导出 ONNX 模型并在 Atlas200DK 上进行 ACL 推理以及通过 Flask Web 接口进行展示结果。
```
### 1. 项目结构
```
DeepTAGE/
├─ checkpoints/                # 模型训练检查点
├─ data/
│  ├─ cifar-10-batches-py/    # CIFAR-10 数据集
│  └─ cifar10_demo_images/     # 测试用示例图片
├─ result/
│  ├─ acc.png                  # 准确率曲线
│  ├─ loss.png                 # 损失曲线
│  └─ train_log.csv            # 训练日志
├─ templates/
│  └─ index.html               # Flask 前端 HTML 页面
├─ acl_infer.py                # Atlas ACL 推理脚本
├─ deeptage.onnx               # 导出的 ONNX 模型
├─ deeptage.py                 # 对文章算法框架的简要实现
├─ export_onnx.py              # 导出 ONNX 脚本
├─ process.py                  # PC 端推理辅助脚本（上传 & 调用板子）
├─ README.md                   # 项目说明文件
├─ server.py                   # Flask 后端服务
├─ show.py                     # 可视化结果脚本
├─ test.py                     # 一些小测试和 CIFAR-10 图片导出脚本
└─ train_cifar10.py            # CIFAR-10 数据集训练脚本
````

### 2. 主要功能
````
1. 模型训练
   - 使用 train_cifar10.py 进行 CIFAR-10 分类模型训练。
   - 支持训练日志保存到 result/train_log.csv，并生成训练曲线 acc.png 和 loss.png。
2. 模型导出  
   - 使用 export_onnx.py 将训练好的模型导出为 deeptage.onnx，便于部署或推理。
3. 推理
   - test.py 支持导出 cifar10 图片用于推理。
   - process.py 支持将图片上传至 Atlas 板子并调用 acl_infer.py 执行 ACL 推理。
4. Atlas ACL 
   - acl_infer.py 用于在 Atlas200DK 上运行模型推理，支持多输出聚合与 Top-5 分类概率返回。
5. Web 前端展示  
   - server.py 启动 Flask 服务，通过 templates/index.html 显示上传图片并返回分类结果。
   - 目前只支持显示最可能分类结果及置信度。
````
### 3. 依赖
````
- Python 3.7+
- Flask
- Pillow
- numpy
- matplotlib
- torchvision（仅训练时需要）
- Atlas200DK + Ascend Toolkit（ACL 推理时）
- 可能还有遗漏，看具体文件报错安装
````
---

## 使用方法

### 1. 训练模型

```bash
python train_cifar10.py
```

* 训练日志和曲线保存在 `result/` 目录，要用可自行查询。

### 2. 导出 ONNX 模型

```bash
python export_onnx.py --input_model checkpoints/best.pth --output_model deeptage.onnx
```
*  将 onnx 模型传上板子
### 3. 转换为om模型
* 在板上运行如下指令将 onnx 模型文件转为 om 模型文件
```bash
atc --model=deeptage.onnx \
    --framework=5 \
    --output=deeptage \
    --input_format=ND \
    --input_shape="input:1,4,3,32,32" \
    --soc_version=Ascend310
```
### 4. Atlas ACL 推理
* 连接板子
* 在PC运行 server.py
```bash
python server.py
```
* 访问 `http://127.0.0.1:5000` 上传图片并查看推理结果。

---

## 注意事项

* ACL 推理依赖 Ascend Toolkit，确保 `PYTHONPATH` 与 `LD_LIBRARY_PATH` 已正确配置。
* 输入图片不必是 CIFAR-10 数据集。
* 推理前一定要确保环境和依赖已经正确配置
* acl_infer.py不用存放在本地，只用放在板子上就可以

---

## 结果可视化

* 训练完成后，可用 `show.py` 对 `result/train_log.csv` 中的训练曲线进行可视化。

---
## 备注

* 实现比较简陋还有很大优化空间

---

