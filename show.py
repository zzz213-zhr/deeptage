import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
log = pd.read_csv("result/train_log.csv", header=None,
                  names=["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5"])

# 绘制 Loss 曲线
plt.figure(figsize=(8,5))
plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
plt.plot(log["epoch"], log["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepTAGE CIFAR-10 Loss Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss.png")

# 绘制 Accuracy 曲线
plt.figure(figsize=(8,5))
plt.plot(log["epoch"], log["train_acc"], label="Train Acc")
plt.plot(log["epoch"], log["val_top1"], label="Val Top1 Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("DeepTAGE CIFAR-10 Accuracy Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("acc.png")

