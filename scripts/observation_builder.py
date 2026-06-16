import numpy as np
import cv2
import torch
from constant import (
    ACTION_ID_TO_NAME,
    PHASE_TO_ONEHOT,
    SEARCH_HINT_TO_ONEHOT,
)

def one_hot_action(action_name: str) -> list[float]:
    vec = [0.0] * len(ACTION_ID_TO_NAME)
    action_id = ACTION_NAME_TO_ID.get(action_name, 0)   # 預設 Hold
    vec[action_id] = 1.0
    return vec

def build_frame_tensor(frame_ring_buffer):
    frames = list(frame_ring_buffer)
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0   # T,H,W,C
    frames = np.transpose(frames, (3,0,1,2))                       # C,T,H,W
    frames_tensor = torch.from_numpy(frames).unsqueeze(0)          # 1,C,T,H,W
    return frames_tensor

def build_extra_tensor(info, pol_state, frame_id_end):
    visible = float(info.get("visible", 0))
    motion = float(info.get("motion", 0.0))

    phase = info.get("phase", "track")
    phase_vec = PHASE_TO_ONEHOT.get(phase, PHASE_TO_ONEHOT["patrol"])

    search_hint = info.get("search_hint", None)
    search_hint_vec = SEARCH_HINT_TO_ONEHOT.get(search_hint, SEARCH_HINT_TO_ONEHOT["none"])

    last_action_name = pol_state.get("last_action", "Hold")
    last_action_vec = one_hot_action(last_action_name)

    last_action_at_frame = pol_state.get("last_action_at_frame", -1)
    if last_action_at_frame < 0:
        time_since_last_action = 0.0
    else:
        time_since_last_action = min(float(frame_id_end - last_action_at_frame) / 10.0, 1.0)

    hold_until_frame = pol_state.get("hold_until_frame", -1)
    hold_active = 1.0 if hold_until_frame > frame_id_end else 0.0
    
    cooldowns = pol_state.get("cooldowns", {})
    evade_ready = 1.0 if frame_id_end >= cooldowns.get("EvadeBack", 0) else 0.0
    turn_ready = 1.0 if frame_id_end >= cooldowns.get("SearchTurn", 0) else 0.0
    patrol_ready = 1.0 if frame_id_end >= cooldowns.get("PatrolStep", 0) else 0.0

    extra = [
        visible,
        motion,
        *phase_vec,
        *search_hint_vec,
        *last_action_vec,
        time_since_last_action,
        hold_active,
        evade_ready,
        turn_ready,
        patrol_ready
    ]
    extra = np.asarray(extra, dtype=np.float32)

    assert extra.shape == (24,), f"extra feature shape mismatch: got {extra.shape}, expected (24,)"
    extra_tensor = torch.from_numpy(extra).unsqueeze(0)   # shape = (1, 24)
    return extra_tensor

ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}