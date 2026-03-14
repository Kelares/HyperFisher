import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import randint
from torch.func import functional_call # <--- The magic import

device = torch.device("cpu")


class HyperNetwork(nn.Module):
	def __init__(self, device, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.layers = nn.Sequential(
			nn.Linear(1, 10),
			nn.ReLU(),
			nn.Linear(10, 17)
		).to(device)

		self.target_network = nn.Sequential(
			nn.Linear(2, 4),
			nn.Tanh(),
			nn.Linear(4, 1)
		).to(device)

		# 🔒 THE SAFETY LOCK: Ensure target network is just a template
		for param in self.target_network.parameters():
			param.requires_grad = False


	def forward(self, task_id, x):
		target_params = self.layers(task_id).squeeze()
		params_dict = self.get_params_dict(target_params)
		return functional_call(self.target_network, params_dict, x)

	def get_params_dict(self, flat_params):
		param_dict = {}
		pointer = 0
		for name, param in self.target_network.named_parameters():
			num_param = param.numel()
			# Grab the slice and reshape it to match the target parameter
			param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
			pointer += num_param
		return param_dict
	
hyper_network = HyperNetwork(device)

# # --- Hypernetwork Initialization Magic ---
# final_layer = hyper_network[-1]

# with torch.no_grad():
#     # 1. Make the hyper-weights tiny so task_id doesn't cause huge initial swings
#     torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
	
#     # 2. Make the hyper-biases act as the random initialization for the target network
#     # 0.1 is a great safe baseline for a small target network
#     torch.nn.init.normal_(final_layer.bias, mean=0.0, std=0.1)


criterion = nn.BCEWithLogitsLoss()
# Swapped to Adam for much faster XOR convergence
optimizer = torch.optim.Adam(hyper_network.parameters(), lr=1e-1)

X_batch = torch.tensor([
	[0.0, 0.0], 
	[0.0, 1.0], 
	[1.0, 0.0], 
	[1.0, 1.0]
], device=device)

Y_batch = torch.tensor([
	[0.0], 
	[1.0], 
	[1.0], 
	[0.0]
], device=device)
# -------------------------

epochs = 100

task_id = torch.tensor([0.0])
total = 0

# Helper function to unpack your 1D vector into a dictionary of differentiable tensors


for epoch in range(epochs):
	hyper_network.train()
	optimizer.zero_grad()
	
	# 1. Generate the flat weights
	output = hyper_network(task_id, X_batch)

	loss = criterion(output, Y_batch)
	loss.backward()
	optimizer.step()
	
	total += loss.item()
	
	# Print less often so we can see the progress clearly
	if (epoch + 1) % 10 == 0:
		print(f"epoch {epoch+1}/{epochs} | avg_loss={total/(epoch+1):.4f} | last_loss={loss.item():.4f}")


# --- EVALUATION ---
print("\n--- Final XOR Test ---")
hyper_network.eval()
with torch.no_grad():
	output = hyper_network(task_id, X_batch)

	
	# Convert raw logits to probabilities (0.0 to 1.0)
	probabilities = torch.sigmoid(output)
	
	# Convert probabilities to crisp 0 or 1 predictions
	predictions = (probabilities >= 0.5).float()
	
	for i in range(4):
		print(f"Input: {X_batch[i].tolist()} -> Raw Logit: {output[i].item():>7.4f} | Prob: {probabilities[i].item():.4f} | Pred: {predictions[i].item()} | Target: {Y_batch[i].item()}")