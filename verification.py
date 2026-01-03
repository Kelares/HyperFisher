import torch
import gymnasium as gym  # Updated to Gymnasium (Modern Standard)
import os

print("✅ 1. Basic Libraries Imported")

# Check GPU
if torch.cuda.is_available():
    print(f"✅ 2. GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 2. No GPU detected. Mamba will be very slow or fail!")

# Check Mamba Installation
try:
    from mamba_ssm import Mamba
    print("✅ 3. Mamba Library Found")
    
    # Quick Test: Create a tiny Mamba block
    model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(1, 10, 16).cuda() # Batch 1, Sequence 10, Dim 16
    y = model(x)
    print("✅ 4. Mamba Forward Pass Successful")
except ImportError:
    print("❌ 3. Mamba not installed. (Did you use Linux/WSL?)")
except Exception as e:
    print(f"❌ 4. Mamba failed to run: {e}")

# Check Offline RL Environment (Gymnasium + Minari)
# Note: D4RL is deprecated for Gymnasium. We check for Minari instead.
try:
    import minari
    print("✅ 5. Minari Library Found (Modern D4RL replacement)")
    
    # Check if we can load a gymnasium environment
    # Note: You might need 'pip install gymnasium[mujoco]'
    try:
        env = gym.make('Hopper-v4') # v4 is the standard Gymnasium Mujoco env
        print(f"✅ 6. Gymnasium Environment Loaded: {env.spec.id}")
    except Exception as env_error:
        print(f"❌ 6. Gymnasium Environment Failed: {env_error}")
        print("   (Try running: pip install gymnasium[mujoco])")

except ImportError:
    print("❌ 5. Minari not installed. (pip install minari)")
    print("   Note: If you strictly need D4RL (old stack), you must use 'gym==0.23.1'")
