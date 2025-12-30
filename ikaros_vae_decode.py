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
        decoded = vae.decode(samples["samples"])
        
        # 确保输出格式正确：移除不必要的维度，并确保数据类型为float32
        # 正确格式应该是(batch_size, height, width, channels)的f32格式
        if len(decoded.shape) == 5:
            decoded = decoded.squeeze(1)  # 移除中间的1维度
        
        # 确保数据类型正确
        decoded = decoded.to(dtype=torch.float32)
        
        # 返回解码后的图像
        return (decoded,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ikaros解码": IkarosVAEDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ikaros解码": "ikaros解码"
}
