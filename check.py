import torch
import torch.nn as nn # Need nn for dtype example

# *** 重要：根据实际位置调整导入语句 ***
from src.CLEAN.model import LayerNormNet1

# --- 配置 ---
# hidden_dim 从错误信息 [..., 512] 推断
hidden_dim_checkpoint = 512
# out_dim 从 checkpoint 和代码中的错误信息推断
output_dim_code = 256
output_dim_checkpoint = 128
# 只需要提供 device 和 dtype 的示例值用于实例化
device_example = 'cpu' # Use CPU for inspection
dtype_example = torch.float32 # Common dtype
# --- 配置结束 ---

print(f"--- 检查模型结构 (hidden={hidden_dim_checkpoint}, out={output_dim_code}) ---")
try:
    # 实例化模型，使用代码当前的输出维度
    model_code = LayerNormNet1(
        hidden_dim=hidden_dim_checkpoint, # Use 512
        out_dim=output_dim_code,          # Use 256
        device=device_example,
        dtype=dtype_example
    )
    print(model_code)
except Exception as e:
    print(f"用 out_dim={output_dim_code} 实例化模型时出错: {e}")
    print("请检查 LayerNormNet1 的 __init__ 方法和提供的示例参数。")

print("\n" + "="*50 + "\n")

print(f"--- 检查模型结构 (hidden={hidden_dim_checkpoint}, out={output_dim_checkpoint}) ---")
try:
    # 实例化模型，使用 checkpoint 文件期望的输出维度
    model_checkpoint = LayerNormNet1(
        hidden_dim=hidden_dim_checkpoint, # Use 512
        out_dim=output_dim_checkpoint,    # Use 128
        device=device_example,
        dtype=dtype_example
    )
    print(model_checkpoint)
except Exception as e:
    print(f"用 out_dim={output_dim_checkpoint} 实例化模型时出错: {e}")
    print("请检查 LayerNormNet1 的 __init__ 方法和提供的示例参数。")