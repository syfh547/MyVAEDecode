import torch
import sys
import os

# 添加ComfyUI到路径
sys.path.append('o:/ComfyUI-aki-v2/ComfyUI')

# 导入自定义节点
try:
    from ikaros_vae_decode import IkarosVAEDecode
    print("成功导入IkarosVAEDecode节点")
    
    # 创建模拟的VAE和潜在样本
    class MockVAE:
        def decode(self, samples):
            # 模拟VAE.decode的返回值
            # 返回形状为(1, 3, 608, 608)的张量
            return torch.randn(1, 3, 608, 608)
    
    # 创建测试数据
    mock_vae = MockVAE()
    mock_samples = {"samples": torch.randn(1, 4, 76, 76)}  # 潜在样本
    
    # 创建节点实例并测试
    node = IkarosVAEDecode()
    result = node.decode(mock_vae, mock_samples)
    
    # 检查输出
    if len(result) > 0:
        image = result[0]
        print(f"测试结果：")
        print(f"  形状: {image.shape}")
        print(f"  数据类型: {image.dtype}")
        print(f"  最小值: {image.min().item()}")
        print(f"  最大值: {image.max().item()}")
        
        # 验证是否符合ComfyUI的IMAGE类型要求
        if len(image.shape) == 4:
            print("✓ 具有正确的4个维度")
        else:
            print("✗ 维度不正确")
            
        if image.dtype == torch.float32:
            print("✓ 数据类型正确(float32)")
        else:
            print("✗ 数据类型不正确")
            
        if image.shape[-1] == 3:
            print("✓ 通道数正确(3)")
        else:
            print("✗ 通道数不正确")
            
        if image.min() >= 0 and image.max() <= 1:
            print("✓ 值范围正确(0-1)")
        else:
            print("✗ 值范围不正确")
    
    print("测试完成！")
    
except Exception as e:
    print(f"测试失败：{e}")
    import traceback
    traceback.print_exc()
