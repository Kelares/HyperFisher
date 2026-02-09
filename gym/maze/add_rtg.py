import numpy as np
import pickle

def load_dataset(file_name):
    with open(file_name, "rb") as f:
        dataset = pickle.load(f)
    return dataset

d = load_dataset("perfect_expert_data_99+.pkl")
#env = gym.make("MiniGrid-MemoryS13Random-v0", render_mode=None)
######################################### env.env.env.env.max_steps
def get_decaying_rtg(rewards, max_steps=845): #FIXED MAX_STEPS FOR MiniGrid-MemoryS13Random-v0
    trajectory_length = len(rewards)
    final_reward = rewards[-1]
    rtg = []
    for t in range(trajectory_length):
        # We calculate the potential reward at time t
        # If final_reward is 0 (fail), we still want the 'pressure' signal
        # so the model learns that time is passing even during failures.
        potential_reward = 1.0 - 0.9 * (t / max_steps)
        
        # If the episode was a success, the RTG is the potential reward.
        # If it was a failure, you can still use the potential_reward 
        # to teach the model 'what it could have had'.
        if final_reward > 0:
            rtg.append(potential_reward)
        else:
            # For failures, we signal that the target is 0, 
            # but we still provide the decay to ground the SSM in time.
            rtg.append(0.0) 
            
    return rtg

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
    rtg = get_decaying_rtg(traj["rewards"])
    traj["rtg"] = rtg
    d[i] = traj

with open("dataset_rtg_99+.pkl", "wb+") as f:
    pickle.dump(d, f)