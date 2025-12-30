import sys
import os

# Add ComfyUI to path
comfy_path = 'o:/ComfyUI-aki-v2/ComfyUI'
sys.path.append(comfy_path)

# Try to import and inspect VAEDecode
try:
    from nodes import VAEDecode
    print("Found VAEDecode in nodes module")
    print(f"VAEDecode class: {VAEDecode}")
    print(f"VAEDecode attributes: {dir(VAEDecode)}")
    
    # Check INPUT_TYPES
    if hasattr(VAEDecode, 'INPUT_TYPES'):
        print(f"INPUT_TYPES: {VAEDecode.INPUT_TYPES}")
    
    # Check FUNCTION
    if hasattr(VAEDecode, 'FUNCTION'):
        print(f"FUNCTION: {VAEDecode.FUNCTION}")
        # Check the method
        if hasattr(VAEDecode, VAEDecode.FUNCTION):
            method = getattr(VAEDecode, VAEDecode.FUNCTION)
            print(f"Method signature: {method.__code__.co_varnames[:method.__code__.co_argcount]}")
            
except ImportError as e:
    print(f"Import error: {e}")
    
    # Try different import paths
    try:
        from comfy.nodes.vae import VAEDecode
        print("Found VAEDecode in comfy.nodes.vae")
    except ImportError:
        print("Could not find VAEDecode")
        
    # List nodes directory
    nodes_dir = os.path.join(comfy_path, 'nodes')
    if os.path.exists(nodes_dir):
        print(f"Nodes directory contents: {os.listdir(nodes_dir)}")
        
        # Check if vae.py exists
        vae_file = os.path.join(nodes_dir, 'vae.py')
        if os.path.exists(vae_file):
            print("Found vae.py in nodes directory")
            with open(vae_file, 'r') as f:
                content = f.read()
                print("First 500 characters of vae.py:")
                print(content[:500])
