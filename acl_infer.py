import sys
import acl
import numpy as np
import ctypes
import json

DEVICE_ID = 0
MODEL_PATH = "/home/HwHiAiUser/deeptage_infer/model/deeptage.om"


# memcpy kind values
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2

def malloc_host_and_copy_from_bytes(py_bytes):
    size = len(py_bytes)
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != 0:
        raise RuntimeError("malloc_host failed, ret=" + str(ret))
    ctypes.memmove(host_ptr, py_bytes, size)
    return host_ptr, size

def copy_device_to_bytes(dev_ptr, size):
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != 0:
        raise RuntimeError("malloc_host failed (copy back), ret=" + str(ret))
    ret = acl.rt.memcpy(host_ptr, size, dev_ptr, size, DEVICE_TO_HOST)
    if ret != 0:
        acl.rt.free_host(host_ptr)
        raise RuntimeError("memcpy DEVICE->HOST failed, ret=" + str(ret))
    data = ctypes.string_at(host_ptr, size)
    acl.rt.free_host(host_ptr)
    return data

def load_bin_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main():
    # 初始化 ACL
    ret = acl.init()
    ret = acl.rt.set_device(DEVICE_ID)
    context, ret = acl.rt.create_context(DEVICE_ID)

    # 加载模型
    model_id, ret = acl.mdl.load_from_file(MODEL_PATH)
    if ret != 0:
        print("Error: load_from_file failed, ret=", ret)
        return

    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)

    # 打印模型期望输入大小
    num_inputs = acl.mdl.get_num_inputs(model_desc)
    print(f"Model has {num_inputs} input(s)")
    for i in range(num_inputs):
        size = acl.mdl.get_input_size_by_index(model_desc, i)
        print(f"Input[{i}] size in bytes: {size}")
    # 如果没有传入 bin 文件，直接退出
    if len(sys.argv) < 2:
        print("No input file provided, only printed model input size. Exiting.")
        # 清理 ACL
        acl.mdl.destroy_desc(model_desc)
        acl.mdl.unload(model_id)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(DEVICE_ID)
        acl.finalize()
        return

    input_path = sys.argv[1]
    print("[Atlas] Load input:", input_path)

    # 准备输入
    input_bytes = load_bin_as_bytes(input_path)
    input_size = len(input_bytes)
    dev_in, ret = acl.rt.malloc(input_size, 0)
    if ret != 0:
        print("malloc dev_in failed, ret=", ret)
        return

    host_in_ptr, _ = malloc_host_and_copy_from_bytes(input_bytes)
    ret = acl.rt.memcpy(dev_in, input_size, host_in_ptr, input_size, HOST_TO_DEVICE)
    acl.rt.free_host(host_in_ptr)
    if ret != 0:
        print("memcpy HOST->DEVICE failed, ret=", ret)
        return

    # 创建输入 dataset
    input_dataset = acl.mdl.create_dataset()
    in_data_buf = acl.create_data_buffer(dev_in, input_size)
    acl.mdl.add_dataset_buffer(input_dataset, in_data_buf)

    # 准备输出
    num_outputs = acl.mdl.get_num_outputs(model_desc)
    output_devs = []
    output_sizes = []
    output_dataset = acl.mdl.create_dataset()
    for i in range(num_outputs):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        output_sizes.append(size_i)
        dev_out_i, ret = acl.rt.malloc(size_i, 0)
        if ret != 0:
            print(f"malloc dev_out[{i}] failed, ret={ret}")
            for p in output_devs:
                try:
                    acl.rt.free(p)
                except Exception:
                    pass
            return
        output_devs.append(dev_out_i)
        out_buf = acl.create_data_buffer(dev_out_i, size_i)
        acl.mdl.add_dataset_buffer(output_dataset, out_buf)

    # 执行推理
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    if ret != 0:
        print("Model execute failed, ret=", ret)
        return

    # 拷贝输出到 host
    outputs = []
    for i, dev_ptr in enumerate(output_devs):
        size_i = output_sizes[i]
        try:
            b = copy_device_to_bytes(dev_ptr, size_i)
        except Exception as e:
            print(f"copy output[{i}] failed:", e)
            b = None
        if b is None:
            outputs.append(None)
            continue
        arr = np.frombuffer(b, dtype=np.float32)
        outputs.append(arr)
        print(f"output[{i}] len={arr.size}")

    # 多输出聚合
    valid_outputs = [o for o in outputs if o is not None]
    if len(valid_outputs) == 0:
        print("No valid outputs retrieved")
        return
    num_classes = valid_outputs[0].size
    stacked = np.stack([o.reshape(num_classes) for o in valid_outputs], axis=0)
    logits = np.mean(stacked, axis=0)
    probs = softmax(logits)
    cls = int(np.argmax(probs))
    score = float(probs[cls])

    # 输出 JSON 方便 PC 解析
    try:
        result = {"class_id": cls, "class_name": cls, "score": round(float(score*100.0), 2)}
        print(json.dumps(result))
    except Exception:
        pass

    print(f"{cls} {score:.4f}")

    # 清理资源
    try:
        acl.mdl.unload(model_id)
        acl.mdl.destroy_desc(model_desc)
    except Exception:
        pass
    for p in output_devs:
        try:
            acl.rt.free(p)
        except Exception:
            pass
    try:
        acl.rt.free(dev_in)
    except Exception:
        pass
    try:
        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)
    except Exception:
        pass
    try:
        acl.rt.destroy_context(context)
        acl.rt.reset_device(DEVICE_ID)
        acl.finalize()
    except Exception:
        pass

if __name__ == "__main__":
    main()
