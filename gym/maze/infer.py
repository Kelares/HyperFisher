from ssm import create_actor, STATE_DIM, ACT_DIM
import torch
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
actor = create_actor(device)

LOSS_ACHIEVED = "0.19779316292089574"
index = 7
FOLDER_PATH = f"runs/{index}_{LOSS_ACHIEVED}"

actor = create_actor(device)

actor.load_state_dict(torch.load(f"{FOLDER_PATH}/agent.pt", map_location=device))
actor.eval()


env = gym.make("MiniGrid-MemoryS13Random-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder=FOLDER_PATH, name_prefix="") # Record every episode
env = FlatObsWrapper(env)


target_return = 1.0 # High goal for Decision Mamba


random_seed = random.randint(0,1_000_000)
random_seed = 297
env.set_wrapper_attr("name_prefix", random_seed)

obs, _ = env.reset(seed=random_seed)
print("SEED: ", env.unwrapped.np_random_seed)
done = False

# Initialize buffers for the sequence
states = torch.from_numpy(obs).float().reshape(1, 1, 2835).to(device)
actions = torch.zeros((1, 1), dtype=torch.long).to(device)
rtgs = torch.tensor([target_return]).float().reshape(1, 1, 1).to(device)

total_reward = 0
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

    # Optional: Keep a sliding window if sequences get too long
    # states = states[:, -30:, :] ...
print(f"Reason {'truncated' if truncated else 'terminated'}")
print(f"Episode finished with reward: {total_reward}")
env.close()