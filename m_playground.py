import torch

# 1. Creation & Basic Arithmetic
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"Addition:\n{A + B}")
print(f"Scalar Mult (A * 10):\n{A * 10}")

# 2. Transpose
print(f"Transpose of A:\n{A.t()}") # Or A.T

# 3. Multiplication: Element-wise vs Matrix Product
# Element-wise (Hadamard)
print(f"Element-wise Mult (A * B):\n{A * B}")

# True Matrix Multiplication (Dot Product)
# Methods: torch.mm(A, B), torch.matmul(A, B), or A @ B
print(f"Matrix Multiplication (A @ B):\n{A @ B}")

# 4. Inversion and Determinant (Linear Algebra)
# Requires square matrices
det = torch.det(A)
inv = torch.inverse(A)
print(f"Determinant of A: {det:.2f}")
print(f"Inverse of A:\n{inv}")

# 5. Reshaping & Squeezing
C = torch.arange(6) # [0, 1, 2, 3, 4, 5]
reshaped = C.view(2, 3) # Becomes a 2x3 matrix
print(f"Reshaped Tensor:\n{reshaped}")

# 6. Reduction (Sum/Mean)
print(f"Sum of A: {A.sum()}")
print(f"Mean of A: {A.mean()}")
