import torch
import numpy as np
import struct
import os

def calculate_quant_params(arr: np.ndarray):
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)

    if abs(max_val - min_val) < 1e-7:
        return 1.0, 0

    scale = (max_val - min_val) / 255.0
    initial_zp = -128.0 - (min_val / scale)
    zero_point = int(np.clip(np.round(initial_zp), -128, 127))

    return scale, zero_point

def quantize_array(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(arr / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)

def write_qbin(filepath: str, tensor_list):
    with open(filepath, "wb") as f:
        magic = 0x514E4554
        f.write(struct.pack("<II", magic, len(tensor_list)))

        for (rows, cols, scale, zp, data_int8) in tensor_list:
            f.write(struct.pack("<IIfi", rows, cols, scale, zp))
            f.write(data_int8.tobytes())

    print(f"Exported {len(tensor_list)} tensors to {filepath}")

def export():
    model_path = "results/mnist_mlp.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run train_model.py first.")

    state_dict = torch.load(model_path, map_location="cpu")

    W1_fp32 = state_dict["fc1.weight"].numpy().T
    b1_fp32 = state_dict["fc1.bias"].numpy().reshape(1, 128)
    W2_fp32 = state_dict["fc2.weight"].numpy().T
    b2_fp32 = state_dict["fc2.bias"].numpy().reshape(1, 10)

    tensors_to_pack = [
        ("W1", W1_fp32),
        ("b1", b1_fp32),
        ("W2", W2_fp32),
        ("b2", b2_fp32)
    ]

    export_records = []
    print("\n--- Quantization Parameters ---")
    for name, arr in tensors_to_pack:
        rows, cols = arr.shape
        scale, zp = calculate_quant_params(arr)
        q_data = quantize_array(arr, scale, zp)
        export_records.append((rows, cols, scale, zp, q_data))
        print(f"Tensor {name:2s} | Shape: {rows:3d}x{cols:3d} | Scale: {scale:.6f} | ZP: {zp:4d}")

    os.makedirs("results", exist_ok=True)
    write_qbin("results/mnist_model.qbin", export_records)

if __name__ == "__main__":
    export()
