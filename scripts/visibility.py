import torch
from constant import ms_per_frame, delta_t_ms

def update(state, frames, pred_name, visible, frame_id_end):
    if state is None:
        state = stateInit()
    phase = ""
    invisible_acc_ms = state["invisible_acc_ms"]
    last_seen_dir = state["last_seen_dir"]
    last_visible_ts_ms = state["last_visible_ts_ms"]
    REACQ_GRACE_MS = 1000
    PATROL_TIMEOUT_MS = 3000
    now_ms = (frame_id_end + 1) * ms_per_frame
    search_hint = None

    mg = motionGate(frames, pred_name, visible)
    motion, stop, pred_name, visible = mg["motion"], mg["stop"], mg["pred_name"], mg["visible"]

    if visible == 1:
        dir_new = estiDirections(frames)
        if dir_new == state["dir_candidate"]:
            state["dir_hysteresis_cnt"] += 1
        else:
            state["dir_candidate"] = dir_new
            state["dir_hysteresis_cnt"] = 1
        if state["dir_hysteresis_cnt"] >= 2:
            last_seen_dir = state["dir_candidate"]
        phase = "track"
        last_visible_ts_ms = now_ms
        invisible_acc_ms = 0
    else :
        invisible_acc_ms += delta_t_ms
        if (now_ms - last_visible_ts_ms) <= REACQ_GRACE_MS:
            phase = "reacq"
        elif invisible_acc_ms > PATROL_TIMEOUT_MS:
            phase = "patrol"
        else:
            phase = "reacq"
    
    if phase == "reacq":
        search_hint = last_seen_dir

    out = {    "visible": visible,                     # 覆寫後
                "phase": phase,
                "search_hint": search_hint,
                "pred_name": pred_name,
                "motion": motion,              # 方便 log/調參
                "invisible_acc_ms": invisible_acc_ms,          # 決策層可看它換策略
                "last_seen_dir": last_seen_dir,
                "last_visible_ts_ms": last_visible_ts_ms}
    
    state["invisible_acc_ms"] = invisible_acc_ms
    state["last_seen_dir"] = last_seen_dir
    state["last_visible_ts_ms"] = last_visible_ts_ms
    state["dir_candidate"] = state.get("dir_candidate", "center")
    state["dir_hysteresis_cnt"] = state.get("dir_hysteresis_cnt", 0)

    return out, state

def stateInit():
    state = {
        "search_hint": None,
        "invisible_acc_ms": 0,
        "last_visible_ts_ms": 0,
        "last_seen_dir": "center",
        "dir_candidate": "center",
        "dir_hysteresis_cnt": 0
    }
    return state
    

def motionGate(frames, pred_name, visible, gate=0.018):
    T = frames.shape[2]

    total_motion = 0.0
    for t in range(T - 1):  # 0..6
        img_hwc      = frames[0, :, t,   :, :].permute(1, 2, 0)  # (H,W,C)
        img_hwc_next = frames[0, :, t+1, :, :].permute(1, 2, 0)  

        img_diff   = torch.abs(img_hwc_next - img_hwc)           # (H,W,C)
        motion_px  = torch.mean(img_diff, dim=2)                 # (H,W)  ← 在每個像素位置上，把 R/G/B 三個通道的差值取平均
        motion_t   = torch.mean(motion_px)                       # 標量，這一對幀的動態強度
        total_motion += motion_t

    motion = (total_motion / (T - 1)).item()                     # 8 幀 → 7 個差分的平均
    stop   = 1 if motion < gate else 0

    if stop == 1:
        pred_name = "none"
        visible   = 0

    print(f"[motion-gate] motion={motion:.5f} gate={gate} stop={stop}")
    return {"motion": motion, "stop": stop, "pred_name": pred_name, "visible": visible}

def estiDirections(frames, kappa=0.07, eps=1e-8):
    frames = frames.squeeze(0) # B,C,T,H,W → C,T,H,W
    T = frames.shape[1]
    W = frames.shape[3]
    x0 = W / 2.0
    total_x_cm = 0
    valid = 0

    for t in range(T - 1):  # 0..6
        img_hwc      = frames[:, t]
        img_hwc_next = frames[:, t+1]
        img_diff   = torch.abs(img_hwc_next - img_hwc)      # (H,W)
        M = img_diff.mean(dim=0)                          # (H,W)

        x_idx = torch.arange(W, device=M.device, dtype=M.dtype)  # 0..W-1
        weights_x = M.sum(dim=0)                                 
        sum_w = float(weights_x.sum().item())
        if sum_w <= eps:
            continue

        x_cm = float((weights_x * x_idx).sum().item()) / (sum_w + eps)  # 標量
        total_x_cm += x_cm
        valid += 1
    
    if valid == 0:
        print("direction=center (no valid motion)")
        return "center"

    total_x_cm /= valid
    offset = (total_x_cm - x0) / W
    direction = ""
    if offset > kappa:
        direction = "right"
    elif offset < -kappa:
        direction = "left"
    else:
        direction = "center"
    print(f"direction={direction}")
    return direction
