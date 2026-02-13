import pickle
import numpy as np

def verify_dataset(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # In your collection script, 'rewards' is a list per episode
    # Success is defined as any reward > 0 in an episode
    episode_returns = [sum(ep['rewards']) for ep in data]
    successes = [1 if r > 0 else 0 for r in episode_returns]
    episode_lengths = [len(ep['observations']) for ep in data]
    # print(f"random rtg: {data[0]['rtg']}")
    print(f"--- Dataset Verification ---")
    print(f"Total Episodes: {len(data)}")
    print(f"Success Rate:   {np.mean(successes) * 100:.2f}%")
    print(f"Average Return: {np.mean(episode_returns):.4f}")
    print(f"Avg Ep Length:  {np.mean(episode_lengths):.2f} steps")
    
    if np.mean(successes) < 0.95:
        print("⚠️ WARNING: Your oracle is not 'expert' yet. Offline RL on noisy data is much harder.")

verify_dataset("MiniGrid-MemoryS9-v0_S9_BIG.pickle")
