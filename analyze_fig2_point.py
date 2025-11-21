import os
import re
import numpy as np
import pandas as pd
import torch


def find_latest_experiment_dir(root="outputs"):
	"""
	Find the most recently modified experiment subdirectory under `root`.
	"""
	if not os.path.isdir(root):
		raise RuntimeError(f"No outputs directory found at: {root}")

	subdirs = [
		os.path.join(root, d)
		for d in os.listdir(root)
		if os.path.isdir(os.path.join(root, d))
	]
	if not subdirs:
		raise RuntimeError(f"No experiment subdirectories found under: {root}")

	return max(subdirs, key=os.path.getmtime)


def compute_F1(exp_dir):
	"""
	Compute forgetting F1 for task 1:
	  F1 = max_t a_{t,1} - a_{T,1}, with accuracies converted from [%] to [0,1].
	"""
	acc_path = os.path.join(exp_dir, "accs.csv")
	if not os.path.isfile(acc_path):
		raise RuntimeError(f"accs.csv not found in: {exp_dir}")

	df = pd.read_csv(acc_path, index_col=0)
	if "1" not in df.columns:
		raise RuntimeError("Task 1 column ('1') not found in accs.csv")

	acc_task1 = df["1"].to_numpy(dtype=float)
	F1 = (acc_task1.max() - acc_task1[-1])
	return float(F1)


def compute_F1_loss(exp_dir):
	"""
	Compute loss-based forgetting for task 1 in the sense of the paper:
	  F1_loss = L_1(w*_2) - L_1(w*_1),
	where w*_1 is the converged parameter after finishing task 1 and w*_2
	is the converged parameter after finishing task 2 (or the final task
	in a 2-task setup). In practice we approximate:
	  L_1(w*_1) by the loss on task 1 at the end of task 1, and
	  L_1(w*_2) by the loss on task 1 at the end of training.
	"""
	loss_path = os.path.join(exp_dir, "loss.csv")
	if not os.path.isfile(loss_path):
		raise RuntimeError(f"loss.csv not found in: {exp_dir}")

	df = pd.read_csv(loss_path, index_col=0)
	if "1" not in df.columns:
		raise RuntimeError("Task 1 column ('1') not found in loss.csv")

	loss_task1 = df["1"].to_numpy(dtype=float)
	# Use Hessian log for task 1 to locate the end-of-task-1 time index.
	hess_path = os.path.join(exp_dir, "hessian_eigs.csv")
	if not os.path.isfile(hess_path):
		raise RuntimeError(f"hessian_eigs.csv not found in: {exp_dir}")

	hess_df = pd.read_csv(hess_path, index_col=0)
	task1_cols = [c for c in hess_df.columns if c.startswith("task-1-epoch-")]
	if not task1_cols:
		raise RuntimeError("No columns for task-1-epoch-* found in hessian_eigs.csv")

	# There is typically a single snapshot for task 1, taken at the end of task 1:
	m = re.search(r"task-1-epoch-(\d+)", task1_cols[0])
	if not m:
		raise RuntimeError(f"Could not parse epoch index from column {task1_cols[0]}")
	idx_end_task1 = int(m.group(1))
	if idx_end_task1 < 0 or idx_end_task1 >= loss_task1.shape[0]:
		raise RuntimeError(
			f"Parsed end-of-task-1 index {idx_end_task1} is out of bounds "
			f"for loss trajectory of length {loss_task1.shape[0]}"
		)

	L1_w1 = loss_task1[idx_end_task1]   # loss on task 1 at end of task 1
	L1_w2 = loss_task1[-1]              # loss on task 1 at end of training

	F1_loss = (L1_w2 - L1_w1)
	return float(F1_loss)


def parse_time_from_model_filename(filename):
	"""
	Extract the integer `time` from a filename like: model-TRIALID-time.pth
	"""
	m = re.search(r"model-[A-Z0-9]+-(\d+)\.pth", filename)
	if not m:
		return None
	return int(m.group(1))


def compute_delta_w_sq(exp_dir):
	"""
	Compute ||Δw||^2 between the first-task and last-task checkpoints.
	We use the earliest and latest `time` indices in the saved model files.
	"""
	model_files = [
		f for f in os.listdir(exp_dir)
		if f.startswith("model-") and f.endswith(".pth")
	]
	if not model_files:
		raise RuntimeError(f"No model-*.pth files found in: {exp_dir}")

	time_to_path = {}
	for f in model_files:
		t = parse_time_from_model_filename(f)
		if t is not None:
			time_to_path[t] = os.path.join(exp_dir, f)

	if not time_to_path:
		raise RuntimeError("Could not parse times from any model-*.pth filenames.")

	t_min = min(time_to_path.keys())
	t_max = max(time_to_path.keys())

	sd1 = torch.load(time_to_path[t_min], map_location="cpu")
	sd2 = torch.load(time_to_path[t_max], map_location="cpu")

	sq_sum = 0.0
	for k in sd1.keys():
		d = (sd2[k] - sd1[k]).view(-1)
		sq_sum += float(d @ d)
	return float(sq_sum)


def load_lambda_max_task1(exp_dir):
	"""
	Extract λ_max for task 1 at the end of task 1 from hessian_eigs.csv.
	We rely on the key pattern used in `log_hessian`:
	  key = 'task-{task_id}-epoch-{time-1}'
	where `time` is the global epoch counter (1, 2, ..., T * E) and the Hessian
	for each task is logged once at the end of that task. For task 1 this
	corresponds to the *largest* epoch index among columns that start with
	'task-1-epoch-'.
	"""
	hess_path = os.path.join(exp_dir, "hessian_eigs.csv")
	if not os.path.isfile(hess_path):
		raise RuntimeError(f"hessian_eigs.csv not found in: {exp_dir}")

	df = pd.read_csv(hess_path, index_col=0)
	# Find all columns corresponding to task 1 Hessian snapshots
	task1_cols = [c for c in df.columns if c.startswith("task-1-epoch-")]
	if not task1_cols:
		raise RuntimeError("No columns for task-1-epoch-* found in hessian_eigs.csv")
	
	# Select the one with the largest epoch index (i.e., end of task 1)
	def _epoch_idx(col_name: str) -> int:
		m = re.search(r"task-1-epoch-(\d+)", col_name)
		return int(m.group(1)) if m else -1
	
	col = max(task1_cols, key=_epoch_idx)
	eigs = df[col].to_numpy(dtype=float)
	if eigs.size == 0:
		raise RuntimeError(f"No eigenvalues stored for column {col}")

	return float(eigs[0])


def main():
	exp_dir = find_latest_experiment_dir(root="outputs")
	print(f"Using experiment directory: {exp_dir}")

	F1 = compute_F1(exp_dir)
	F1_loss = compute_F1_loss(exp_dir)
	delta_w_sq = compute_delta_w_sq(exp_dir)
	lambda_max_1 = load_lambda_max_task1(exp_dir)

	x = lambda_max_1 * delta_w_sq * 0.01
	y = F1

	print(f"F1 (forgetting for task 1)        = {y:.6f}")
	print(f"F1_loss (loss-based forgetting)  = {F1_loss:.6f}")
	print(f"||Δw||^2 (between task1 & last) = {delta_w_sq:.6f}")
	print(f"λ_max^(1)                         = {lambda_max_1:.6f}")
	print(f"(x, y) = (λ_max^(1)*||Δw||^2, F1) = ({x:.6f}, {y:.6f})")


if __name__ == "__main__":
	main()

