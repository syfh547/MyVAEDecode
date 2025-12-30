import sys
import os

# 添加ComfyUI到路径
sys.path.append('o:/ComfyUI-aki-v2/ComfyUI')

try:
    # 尝试导入VAE相关模块
    from nodes import VAEDecode
    print("成功导入VAEDecode节点")
    
    # 查看VAEDecode的实现
    print("\nVAEDecode类信息:")
    print(f"FUNCTION: {VAEDecode.FUNCTION}")
    print(f"INPUT_TYPES: {VAEDecode.INPUT_TYPES}")
    print(f"RETURN_TYPES: {VAEDecode.RETURN_TYPES}")
    
    # 查看解码方法
    if hasattr(VAEDecode, VAEDecode.FUNCTION):
        method = getattr(VAEDecode, VAEDecode.FUNCTION)
        print(f"\n解码方法: {VAEDecode.FUNCTION}")
        print(f"方法签名: {method.__code__.co_varnames[:method.__code__.co_argcount]}")
        
        # 尝试获取方法的源代码
        import inspect
        source = inspect.getsource(method)
        print(f"\n方法源代码:\n{source}")
        
except Exception as e:
    print(f"导入错误: {e}")
    
    # 尝试不同的导入路径
    try:
        # 尝试直接导入VAE类
        from comfy.sd import VAE
        print("\n成功导入VAE类")
        print(f"VAE类属性: {[attr for attr in dir(VAE) if not attr.startswith('_')]}")
        
        # 查看是否有decode相关方法
        decode_methods = [attr for attr in dir(VAE) if 'decode' in attr.lower()]
        print(f"\nVAE类中的解码方法: {decode_methods}")
        
    except Exception as e2:
        print(f"导入VAE类错误: {e2}")
        
        # 查看nodes目录下的文件
        nodes_dir = os.path.join('o:/ComfyUI-aki-v2/ComfyUI', 'nodes')
        if os.path.exists(nodes_dir):
            print(f"\nNodes目录内容: {os.listdir(nodes_dir)}")
            
            # 尝试查看vae.py文件内容
            vae_file = os.path.join(nodes_dir, 'vae.py')
            if os.path.exists(vae_file):
                print(f"\n查看vae.py文件...")
                with open(vae_file, 'r') as f:
                    content = f.read(2000)  # 只读取前2000个字符
                    print(content)
