from ssm import create_actor, STATE_DIM, ACT_DIM
import torch
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from pathlib import Path
import os
import re
import numpy as np


LOSS_ACHIEVED = "0.20568031128396325"
index = 5
FOLDER_PATH = f"runs/{index}_{LOSS_ACHIEVED}"

device = "cuda" if torch.cuda.is_available() else "cpu"
actor = create_actor(device)

actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device))
actor.eval()


env = gym.make("MiniGrid-MemoryS13Random-v0", render_mode="rgb_array")

env = FlatObsWrapper(env)


NUM_EPISODES = 100
SEEDS = [200 + i for i in range(NUM_EPISODES)]
record = {"seeds": {}}
env.close()
target_return = 1.0

seed_rewards = []
succesful_runs = 0
for seed in SEEDS:
    record["seeds"][seed] = {}

    obs, _ = env.reset(seed=seed) 
    done = False
    total_reward = 0

    # Initialize buffers for the sequence
    states = torch.from_numpy(obs).float().reshape(1, 1, 2835).to(device)
    actions = torch.zeros((1, 1), dtype=torch.long).to(device)
    rtgs = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)


    while not done:
        with torch.no_grad():
            # Get action prediction from Mamba
            logits = actor(states, actions, rtgs)
            action = torch.argmax(logits[:, -1, :], dim=-1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated

        total_reward += reward
        
        # Update sequence buffers (Decision Mamba needs the history)
        cur_state = torch.from_numpy(obs).float().reshape(1, 1, 2835).to(device)
        cur_rtg = torch.tensor([target_return - total_reward]).float().reshape(1, 1, 1).to(device)
        cur_act = torch.tensor([[action]]).to(device)

        states = torch.cat([states, cur_state], dim=1)
        actions = torch.cat([actions, cur_act], dim=1)
        rtgs = torch.cat([rtgs, cur_rtg], dim=1)
    success = total_reward
    record["seeds"][seed] = {
        "reward": total_reward,
        "steps": len(actions[0]),
        "reason": 'Success' if total_reward else 'Fail'
    }
    succesful_runs += 1 if total_reward else 0
    print(f"Reason {'truncated' if truncated else 'terminated'}")
    print(f"Episode finished with reward: {total_reward}")
    seed_rewards.append(reward)

env.close()

run_dir = Path(FOLDER_PATH) / "benchmarks"
run_dir.mkdir(parents=True, exist_ok=True)


mean_score = np.mean(seed_rewards)
std_score = np.std(seed_rewards)
min_score = np.min(seed_rewards)
max_score = np.max(seed_rewards)

with open(f"{run_dir}/benchmark.txt", "w+") as f:
    f.write("="*30 +"\n")
    f.write(f"📊 FINAL RESULTS ({NUM_EPISODES} Episodes)\n")
    f.write(f"Mean: {mean_score:.2f} ± {std_score:.2f}\n")
    f.write(f"Range: [{min_score:.2f}, {max_score:.2f}]\n")
    f.write(f"success rate: {(succesful_runs/(len(SEEDS)))*100}%\n")
    f.write("RECORDS:::\n")
    for seed in record["seeds"]:
        f.write("="*30 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"  reward: {record['seeds'][seed]['reward']}\n")
        f.write(f"  steps_taken: {record['seeds'][seed]['steps']}\n")
        f.write(f"  reason: {record['seeds'][seed]['reason']}\n")
        f.write("="*30 + "\n")

record["Result"] = {
    "Number of episodes": NUM_EPISODES, 
    "Mean": f"{mean_score:.2f} ± {std_score:.2f}", 
    "Range": f"[{min_score:.2f}, {max_score:.2f}]",
    "Success rate": f"{(succesful_runs/(len(SEEDS)))*100}%"
}
