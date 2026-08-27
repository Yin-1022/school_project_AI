import numpy as np

from constant import ACTION_ID_TO_NAME
from policy import is_ready
from action_mask import build_action_mask

ACTION_NAME_TO_ID = {
    name: action_id
    for action_id, name in ACTION_ID_TO_NAME.items()
}

def case_1_checking():
    # Case 1: All actions are ready & Non-visible Track
    pol_state = {
        "cooldowns": {
            "EvadeBack": 0,
            "SearchTurn": 0,
            "PatrolStep": 0,
        }
    }
    frame_id_end = 100
    info = {
        "visible": 0,
        "phase": "reacq",
    }

    mask = build_action_mask(pol_state, frame_id_end, info)
    expected = np.ones(10, dtype=np.bool_)
    assert np.array_equal(mask, expected)
    assert mask.shape == (10,)
    assert mask.dtype == np.bool_
    assert np.all(mask)

    print("all-ready mask: OK")

def case_2_checking():
    # Case 2: EvadeBack is not ready
    pol_state = {
        "cooldowns": {
            "EvadeBack": 120,
            "SearchTurn": 0,
            "PatrolStep": 0,
        }
    }
    frame_id_end = 100
    info = {
        "visible": 0,
        "phase": "reacq",
    }

    mask = build_action_mask(pol_state, frame_id_end, info)
    expected = np.ones(10, dtype=np.bool_)
    expected[ACTION_NAME_TO_ID["EvadeBack"]] = False  # EvadeBack is not ready

    assert np.array_equal(mask, expected)
    assert mask.shape == (10,)
    assert mask.dtype == np.bool_
    assert not mask[ACTION_NAME_TO_ID["EvadeBack"]]

    print("Evade not-ready mask: OK")

def case_3_checking():
    # Case 3: All actions are not ready
    pol_state = {
        "cooldowns": {
            "EvadeBack": 120,
            "SearchTurn": 120,
            "PatrolStep": 120,
        }
    }
    frame_id_end = 100
    info = {
        "visible": 0,
        "phase": "reacq",
    }

    mask = build_action_mask(pol_state, frame_id_end, info)
    expected = np.ones(10, dtype=np.bool_)
    expected[ACTION_NAME_TO_ID["EvadeBack"]] = False
    expected[ACTION_NAME_TO_ID["SearchTurnLeft"]] = False
    expected[ACTION_NAME_TO_ID["SearchTurnRight"]] = False
    expected[ACTION_NAME_TO_ID["PatrolStepLeft"]] = False
    expected[ACTION_NAME_TO_ID["PatrolStepRight"]] = False

    assert np.array_equal(mask, expected)
    assert mask.shape == (10,)
    assert mask.dtype == np.bool_
    assert not mask[ACTION_NAME_TO_ID["EvadeBack"]]
    assert not mask[ACTION_NAME_TO_ID["SearchTurnLeft"]]
    assert not mask[ACTION_NAME_TO_ID["SearchTurnRight"]]
    assert not mask[ACTION_NAME_TO_ID["PatrolStepLeft"]]
    assert not mask[ACTION_NAME_TO_ID["PatrolStepRight"]]

    print("All not-ready mask: OK")

def case_4_checking():
    # Case 4: visible=1 + track
    pol_state = {
        "cooldowns": {
            "EvadeBack": 0,
            "SearchTurn": 0,
            "PatrolStep": 0,
        }
    }
    frame_id_end = 100
    info = {
        "visible": 1,
        "phase": "track",
    }

    mask = build_action_mask(pol_state, frame_id_end, info)
    expected = np.ones(10, dtype=np.bool_)
    expected[ACTION_NAME_TO_ID["SearchTurnLeft"]] = False
    expected[ACTION_NAME_TO_ID["SearchTurnRight"]] = False
    expected[ACTION_NAME_TO_ID["PatrolStepLeft"]] = False
    expected[ACTION_NAME_TO_ID["PatrolStepRight"]] = False

    assert np.array_equal(mask, expected)
    assert mask.shape == (10,)
    assert mask.dtype == np.bool_
    assert not mask[ACTION_NAME_TO_ID["SearchTurnLeft"]]
    assert not mask[ACTION_NAME_TO_ID["SearchTurnRight"]]
    assert not mask[ACTION_NAME_TO_ID["PatrolStepLeft"]]
    assert not mask[ACTION_NAME_TO_ID["PatrolStepRight"]]
    assert mask[ACTION_NAME_TO_ID["EvadeBack"]]
    assert mask[ACTION_NAME_TO_ID["Retreat"]]

    print("visible-track semantic mask: OK")

def case_5_checking():
    # Case 5: At least one valid action
    pol_state = {
        "cooldowns": {
            "EvadeBack": 120,
            "SearchTurn": 120,
            "PatrolStep": 120,
        }
    }
    frame_id_end = 100
    info = {
        "visible": 1,
        "phase": "track",
    }

    mask = build_action_mask(
        pol_state,
        frame_id_end,
        info,
    )

    assert mask.any()
    print("At least one valid action: OK")

def main():
    case_1_checking()
    case_2_checking()
    case_3_checking()
    case_4_checking()
    case_5_checking()

    print("Action mask: OK")

if __name__ == "__main__":
    main()