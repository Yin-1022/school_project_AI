from pathlib import Path

#For extract_clips.py
CLIP_FRAMES     = 8          
CLIP_STRIDE     = 4          
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
RAW_DIR         = Path("data/raw_videos")
OUT_DIR         = Path("data/clips")
ms_per_frame    = int(1000 / TARGET_FPS)
delta_t_ms      = CLIP_STRIDE * ms_per_frame

TAU_ACTION = 0.35
MIN_HOLD_FRAMES = 2
RT_FRAMES = round(0.2 * TARGET_FPS)
CD_EVADE = round(0.8 * TARGET_FPS)
CD_TURN = round(0.3 * TARGET_FPS)
CD_PATROL = round(0.5 * TARGET_FPS)

ACTION_ID_TO_NAME: dict[int, str] = {
    0: "Hold",
    1: "Advance",
    2: "StrafeLeft",
    3: "StrafeRight",
    4: "EvadeBack",
    5: "Retreat",
    6: "SearchTurnLeft",
    7: "SearchTurnRight",
    8: "PatrolStepLeft",
    9: "PatrolStepRight",
}

obs_features = {
    "frames": {
        "shape": (3, 8, 192, 192),
        "dtype": "float32",
        "range": [0.0, 1.0]
    },
    "extra": {
        "shape": (24,),
        "dtype": "float32",
        "features": {
            "visible": {
                "type": "scalar",
                "values": [0.0, 1.0]
            },
            "motion": {
                "type": "scalar",
            },
            "phase": {
                "type": "one_hot",
                "size": 3,
                "mapping": {
                    "track":  [1, 0, 0],
                    "reacq":  [0, 1, 0],
                    "patrol": [0, 0, 1]
                }
            },
            "search_hint": {
                "type": "one_hot",
                "size": 4,
                "mapping": {
                    "left":   [1, 0, 0, 0],
                    "right":  [0, 1, 0, 0],
                    "center": [0, 0, 1, 0],
                    "none":   [0, 0, 0, 1]
                }
            },
            "last_action": {
                "type": "one_hot",
                "size": 10,
                "mapping": {
                    "Hold":             [1,0,0,0,0,0,0,0,0,0],
                    "Advance":          [0,1,0,0,0,0,0,0,0,0],
                    "StrafeLeft":       [0,0,1,0,0,0,0,0,0,0],
                    "StrafeRight":      [0,0,0,1,0,0,0,0,0,0],
                    "EvadeBack":        [0,0,0,0,1,0,0,0,0,0],
                    "Retreat":          [0,0,0,0,0,1,0,0,0,0],
                    "SearchTurnLeft":   [0,0,0,0,0,0,1,0,0,0],
                    "SearchTurnRight":  [0,0,0,0,0,0,0,1,0,0],
                    "PatrolStepLeft":   [0,0,0,0,0,0,0,0,1,0],
                    "PatrolStepRight":  [0,0,0,0,0,0,0,0,0,1]
                }
            },
            "time_since_last_action": {
                "type": "scalar",
            },
            "hold_active": {
                "type": "scalar",
                "values": [0.0, 1.0]
            },
            "evade_ready": {
                "type": "scalar",
                "values": [0.0, 1.0]
            },
            "turn_ready": {
                "type": "scalar",
                "values": [0.0, 1.0]
            },
            "patrol_ready": {
                "type": "scalar",
                "values": [0.0, 1.0]
            }
        }
    }
}

PHASE_TO_ONEHOT = {
    "track":  [1.0, 0.0, 0.0],
    "reacq":  [0.0, 1.0, 0.0],
    "patrol": [0.0, 0.0, 1.0],
}

SEARCH_HINT_TO_ONEHOT = {
    "left":   [1.0, 0.0, 0.0, 0.0],
    "right":  [0.0, 1.0, 0.0, 0.0],
    "center": [0.0, 0.0, 1.0, 0.0],
    "none":   [0.0, 0.0, 0.0, 1.0],
    None:     [0.0, 0.0, 0.0, 1.0],
}

extra_feature_order = [
    "visible",
    "motion",
    "phase_track",
    "phase_reacq",
    "phase_patrol",
    "search_left",
    "search_right",
    "search_center",
    "search_none",
    "last_action_Hold",
    "last_action_Advance",
    "last_action_StrafeLeft",
    "last_action_StrafeRight",
    "last_action_EvadeBack",
    "last_action_Retreat",
    "last_action_SearchTurnLeft",
    "last_action_SearchTurnRight",
    "last_action_PatrolStepLeft",
    "last_action_PatrolStepRight",
    "time_since_last_action",
    "hold_active",
    "evade_ready",
    "turn_ready",
    "patrol_ready",
]

