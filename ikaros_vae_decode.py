import torch
import numpy as np
from PIL import Image
import folder_paths

# 定义ikaros解码节点
class IkarosVAEDecode:
    # 节点分类，使用用户指定的分类名
    CATEGORY = "💗ikaros节点"
    
    # 节点的主要功能函数
    FUNCTION = "decode"
    
    # 定义输入参数
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
        }
    
    # 定义输出类型
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    # 解码函数实现
    def decode(self, vae, samples):
        # 使用VAE的decode方法解码潜在空间
        result = vae.decode(samples["samples"])
        
        # 检查decode结果是否为元组
        if isinstance(result, tuple):
            # 如果是元组，取第一个元素
            decoded = result[0]
        else:
            decoded = result
        
        # 确保输出格式完全符合ComfyUI IMAGE类型要求：
        # 1. 确保是torch张量
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.tensor(decoded)
        
        # 2. 强制转换为float32（第一步就处理数据类型）
        decoded = decoded.to(dtype=torch.float32)
        
        # 3. 处理维度结构
        if len(decoded.shape) == 5:
            # 情况1: 5D张量 (可能是视频序列: batch, frames, height, width, channels)
            # 将frames维度合并到batch维度
            batch_size, frames, height, width, channels = decoded.shape
            decoded = decoded.view(batch_size * frames, height, width, channels)
        elif len(decoded.shape) == 4:
            # 情况2: 4D张量
            if decoded.shape[1] == 3:  # (batch, channels, height, width)
                decoded = decoded.permute(0, 2, 3, 1)  # 转换为(batch, height, width, channels)
            elif decoded.shape[3] == 3:  # 已经是正确格式
                pass
            
            # 显式移除每个位置上大小为1的维度
            new_shape = []
            for i, dim in enumerate(decoded.shape):
                if dim != 1 or i == 0:  # 保留批次维度，即使大小为1
                    new_shape.append(dim)
            
            # 如果形状改变了，重新塑形
            if len(new_shape) != 4:
                decoded = decoded.view(new_shape)
        elif len(decoded.shape) == 3:
            # 情况2: 3D张量
            if decoded.shape[0] == 3:  # (channels, height, width)
                decoded = decoded.permute(1, 2, 0)  # 转换为(height, width, channels)
            decoded = decoded.unsqueeze(0)  # 添加批次维度，变为4D
        elif len(decoded.shape) == 2:
            # 情况3: 2D张量
            decoded = decoded.unsqueeze(0).unsqueeze(-1)  # 转换为(batch, height, width, 1)
        
        # 4. 最后确保4个维度
        if len(decoded.shape) != 4:
            raise ValueError(f"Unexpected tensor shape: {decoded.shape}. Expected (batch_size, height, width, channels)")
        
        # 5. 确保通道数为3
        if decoded.shape[-1] == 1:
            decoded = decoded.repeat(1, 1, 1, 3)  # 单通道转RGB
        elif decoded.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {decoded.shape[-1]}. Expected 3")
        
        # 6. 确保值范围在0-1之间
        decoded = torch.clamp(decoded, 0.0, 1.0)
        
        # 最终验证：确保形状是(batch, height, width, 3)且数据类型是float32
        assert len(decoded.shape) == 4, f"Final shape must be 4D, got {decoded.shape}"
        assert decoded.shape[-1] == 3, f"Final must have 3 channels, got {decoded.shape[-1]}"
        assert decoded.dtype == torch.float32, f"Final dtype must be float32, got {decoded.dtype}"
        
        # 返回解码后的图像
        return (decoded,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ikaros解码": IkarosVAEDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ikaros解码": "ikaros解码"
}
