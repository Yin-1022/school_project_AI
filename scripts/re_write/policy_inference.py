import torch
from models import TeacherPolicyNet
from constant import (
    ACTION_ID_TO_NAME,
)

def load_model(weights_path:str, device:str ="cuda"):
    model = TeacherPolicyNet(in_ch=3, extra_dim=24, num_actions=10)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def infer_action(frames, extra, model):
    device = next(model.parameters()).device
    frames = frames.to(device)
    extra = extra.to(device)
    
    with torch.no_grad():
        logits = model(frames, extra)
        probs = torch.softmax(logits, dim=1)

    action_id = probs.argmax(dim=1).item()
    conf = probs[0, action_id].item()
    action_name = ACTION_ID_TO_NAME[action_id]
    topk_probs, topk_ids = torch.topk(probs, k=3, dim=1)

    return {
        "action_id": action_id,
        "action_name": action_name,
        "conf": conf,
        "logits": logits,
        "probs": probs,
        "topk_ids": topk_ids.cpu().numpy(),
        "topk_probs": topk_probs.cpu().numpy(),
    }