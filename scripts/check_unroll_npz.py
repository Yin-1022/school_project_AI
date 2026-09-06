import numpy as np

path = "data/rollouts/rollouts_bc_v2/rollout_v3_train_masked_1788680369356.npz"

with np.load(path, allow_pickle=False) as data:
    print(data.files)

    print("frames:", data["frames"].shape, data["frames"].dtype)
    print("bootstrap_frames:", data["bootstrap_frames"].shape, data["bootstrap_frames"].dtype)
    print("bootstrap_extra:", data["bootstrap_extra"].shape)
    print("rollout_profile:", data["rollout_profile"].item())
    print("done sum:", data["done"].sum())