import torch
from deeptage import SimpleSpikingNet

# 加载模型
model = SimpleSpikingNet(in_channels=3, num_classes=10, channels=[32,64,128], T=4)
checkpoint = torch.load("checkpoints/model_best.pth.tar", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 构造 dummy 输入
dummy = torch.randn(1, 4, 3, 32, 32)

# 导出 ONNX
torch.onnx.export(
    model,
    dummy,
    "deeptage.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True
)

print("导出成功：deeptage.onnx")
