import torch

learner = torch.load(
    "data/meta/impala_learner_smoke.pt",
    map_location="cpu",
)

actor = torch.load(
    "data/meta/impala_actor_smoke.pt",
    map_location="cpu",
)

print("Learner:", learner.keys())
print("Actor:", actor.keys())

print("Learner step:", learner["training_step"])
print("Actor step:", actor["training_step"])