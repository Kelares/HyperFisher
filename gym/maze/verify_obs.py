# In a separate script
import numpy as np
import pickle

# Load Dataset Frame
# with open("dataset_big.pickle", "rb") as f:
#     data = pickle.load(f)
# ds_frame = data[0]['observations'][0] # First frame of first ep

# Load Inference Frame (Save this from your infer.py loop)
# np.save("infer_frame.npy", obs)
# inf_frame = np.load("infer_frame.npy")
inf_frame = np.load("infer_frame.npy")
print("infer: ", inf_frame)

train_frame = np.load("train_frame.npy")
print("train: ", train_frame)
# 1. Check Stats
print(f"train Mean: {train_frame.mean()}, Inf Mean: {inf_frame.mean()}")
print(f"train Shape: {train_frame.shape}, Inf Shape: {inf_frame.shape}")

# 2. Check Visual Difference
diff = np.abs(train_frame - inf_frame)
print(f"Max Difference: {diff.max()}") # Should be 0.0 or extremely close (1e-7)


# print(f"DS Mean: {ds_frame.mean()}, Inf Mean: {inf_frame.mean()}")
# print(f"DS Shape: {ds_frame.shape}, Inf Shape: {inf_frame.shape}")

# # 2. Check Visual Difference
# diff = np.abs(ds_frame - inf_frame)
# print(f"Max Difference: {diff.max()}") # Should be 0.0 or extremely close (1e-7)