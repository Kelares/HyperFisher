import numpy as np
import pickle

def load_dataset(file_name):
    with open(file_name, "rb") as f:
        dataset = pickle.load(f)
    return dataset

d = load_dataset("perfect_expert_data.pkl")


def calculate_rtg(rewards, gamma=1.0):
    """
    Calculates the cumulative sum of future rewards.
    rewards: list or np.array of rewards in an episode
    """
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return += rewards[t]
        rtg[t] = running_return
    return rtg

for i, traj in enumerate(d):
    rtg = calculate_rtg(traj["rewards"])
    traj["rtg"] = rtg
    d[i] = traj

with open("perfect_expert_data+.pkl", "wb+") as f:
    pickle.dump(d, f)