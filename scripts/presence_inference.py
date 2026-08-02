import torch

from presence_model import PresenceNet

def load_presence_model(
    checkpoint_path: str,
    device: str,
) -> PresenceNet:
    model = PresenceNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[presence] loaded checkpoint={checkpoint_path} "
        f"epoch={checkpoint.get('epoch', 'unknown')}"
    )

    return model


@torch.no_grad()
def infer_player_presence(
    frames: torch.Tensor,
    model: PresenceNet,
) -> float:
    """
    frames:
      (1, 3, 8, 192, 192)
      或 (3, 8, 192, 192)

    return:
      玩家存在機率 0~1
    """
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)

    device = next(model.parameters()).device

    frames = frames.to(
        device=device,
        dtype=torch.float32,
    )

    logits = model(frames)
    probability = torch.sigmoid(logits)

    return float(probability[0].item())