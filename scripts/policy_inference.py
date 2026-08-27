import torch
from models import TeacherPolicyNet, Small3DNet
from constant import (
    ACTION_ID_TO_NAME,
)
from read_datasets import CLASS_TO_ID

def load_model(weights_path:str, device:str ="cuda"):
    model = TeacherPolicyNet(in_ch=3, extra_dim=24, num_actions=10)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def infer_action(frames, extra, model, sample=False, action_mask=None):
    device = next(model.parameters()).device
    frames = frames.to(device)
    extra = extra.to(device)
    
    with torch.no_grad():
        logits = model(frames, extra)

        if action_mask is not None:
            masked_logits = apply_action_mask(logits, action_mask)
        else:
            masked_logits = logits

        probs = torch.softmax(masked_logits, dim=1)

    if sample:
        action_id = torch.multinomial(probs, num_samples=1).item()
    else:
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
        # "value": value,
    }

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

def load_action_cls_model(weights_path: str, device: str = "cuda"):
    model = Small3DNet(in_ch=3, num_classes=len(CLASS_TO_ID))
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def infer_player_state(frames, model):
    device = next(model.parameters()).device
    frames = frames.to(device)

    with torch.no_grad():
        logits = model(frames)
        probs = torch.softmax(logits, dim=1)

    pred_id = probs.argmax(dim=1).item()
    conf = probs[0, pred_id].item()
    pred_name = ID_TO_CLASS[pred_id]

    topk_probs, topk_ids = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)

    return {
        "pred_id": pred_id,
        "pred_name": pred_name,
        "conf": conf,
        "logits": logits,
        "probs": probs,
        "topk_ids": topk_ids.cpu().numpy(),
        "topk_probs": topk_probs.cpu().numpy(),
    }

def apply_action_mask(logits, action_mask):
    mask_tensor = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=logits.device,
    )

    while mask_tensor.ndim < logits.ndim:
        mask_tensor = mask_tensor.unsqueeze(0)

    return logits.masked_fill(
        ~mask_tensor,
        float("-inf"),
    )