import numpy as np

from constant import ACTION_ID_TO_NAME
from policy import is_ready

ACTION_NAME_TO_ID = {
    name: action_id
    for action_id, name in ACTION_ID_TO_NAME.items()
}

def build_action_mask(pol_state, frame_id_end, info):
    mask = np.ones(len(ACTION_ID_TO_NAME), dtype=np.bool_,)

    if not is_ready(pol_state, "EvadeBack", frame_id_end):
        mask[ACTION_NAME_TO_ID["EvadeBack"]] = False

    if not is_ready(pol_state, "SearchTurn", frame_id_end):
        mask[ACTION_NAME_TO_ID["SearchTurnLeft"]] = False
        mask[ACTION_NAME_TO_ID["SearchTurnRight"]] = False

    if not is_ready(pol_state, "PatrolStep", frame_id_end):
        mask[ACTION_NAME_TO_ID["PatrolStepLeft"]] = False
        mask[ACTION_NAME_TO_ID["PatrolStepRight"]] = False

    visible = info.get("visible", 0)
    phase = info.get("phase", "patrol")

    if visible == 1 and phase == "track":
        mask[ACTION_NAME_TO_ID["SearchTurnLeft"]] = False
        mask[ACTION_NAME_TO_ID["SearchTurnRight"]] = False
        mask[ACTION_NAME_TO_ID["PatrolStepLeft"]] = False
        mask[ACTION_NAME_TO_ID["PatrolStepRight"]] = False

    if not mask.any():
        raise RuntimeError(
            "Action mask contains no valid actions"
        )

    return mask