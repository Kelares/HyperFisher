import torch
import gym
import d4rl

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

# Check D4RL Environment
try:
    env = gym.make('hopper-medium-v2')
    dataset = env.get_dataset()
    print(f"✅ 5. D4RL Dataset Loaded: {dataset['observations'].shape} samples found")
except Exception as e:
    print(f"❌ 5. D4RL Failed: {e}")
