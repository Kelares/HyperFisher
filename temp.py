import argparse
parser = argparse.ArgumentParser(description="Continual Learning Experiments CLI")

parser.add_argument("--seed", type=int, default=1234)

args = parser.parse_args()

for arg in args.keys():
	print(arg)
