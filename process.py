import tempfile
import subprocess
import json
import numpy as np
from PIL import Image
import io

# CIFAR-10 类别名称
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Atlas 板子信息
ATLAS_IP = "192.168.137.2"
ATLAS_DIR = "/home/HwHiAiUser/deeptage_infer"

# CIFAR-10 训练时的归一化
MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1,1,3)
STD  = np.array([0.247, 0.243, 0.261]).reshape(1,1,3)

def preprocess_image(image_bytes, T=1):
    """
    将上传的图片转成符合 Atlas 输入的 float32 bin
    使用训练时的标准化
    输出 shape: (1,T,C,H,W) if T>1 else (1,C,H,W)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32), Image.BICUBIC)  # 高质量缩放

    arr = np.array(img).astype(np.float32) / 255.0  # [H,W,C]
    arr = (arr - MEAN) / STD  # 标准化
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW

    if T > 1:
        arr_t = np.stack([arr] * T, axis=0)  # (T,C,H,W)
        arr_batched = np.expand_dims(arr_t, axis=0)  # (1,T,C,H,W)
    else:
        arr_batched = np.expand_dims(arr, axis=0)  # (1,C,H,W)

    return arr_batched.astype(np.float32)

def run_acl_infer(image_bytes, T=1):
    arr = preprocess_image(image_bytes, T=T)

    # 生成本地临时 bin 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        arr.tofile(f)
        local_bin = f.name

    print(f"本地临时文件： {local_bin}")

    # 1. 上传到板子
    scp_cmd = [
        "scp",
        local_bin,
        f"HwHiAiUser@{ATLAS_IP}:{ATLAS_DIR}/input.bin"
    ]
    print("上传命令：", " ".join(scp_cmd))
    subprocess.check_call(scp_cmd)

    # 2. 调用板子推理
    ssh_cmd = [
        "ssh",
        f"HwHiAiUser@{ATLAS_IP}",
        (
            "export PYTHONPATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH && "
            "export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH && "
            f"cd {ATLAS_DIR} && python3 acl_infer.py {ATLAS_DIR}/input.bin"
        )
    ]
    print("执行推理命令：", " ".join(ssh_cmd))
    out = subprocess.check_output(ssh_cmd).decode("utf-8")
    print("Atlas 输出：", out)

    # 3. 找到 JSON 行
    json_line = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break

    if not json_line:
        raise ValueError("未找到有效的 JSON 推理结果！")

    data = json.loads(json_line)

    class_id = int(data["class_id"])
    score = float(data["score"])

    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "score": score
    }

# 测试示例
if __name__ == "__main__":
    with open("test.jpg", "rb") as f:
        image_bytes = f.read()
    # 可以传 T=1 或 T>1
    result = run_acl_infer(image_bytes, T=1)
    print(result)
