import numpy as np

from action_postprocess import simulate_action_with_state
from constant import ACTION_ID_TO_NAME
from observation_builder import ACTION_NAME_TO_ID

def compute_behavior_probs(prob, pol_state, topk_actions, frame_id_end, info):
    num_actions = len(ACTION_ID_TO_NAME)
    behavior_probs = np.zeros(num_actions, dtype=np.float32)
    action_mapping = {}

    for proposed_action_id in range(num_actions):
        proposed_action_name = ACTION_ID_TO_NAME[proposed_action_id]

        final_action, _, _ = simulate_action_with_state(
            pol_state=pol_state,
            proposed_action=proposed_action_name,
            topk_actions=topk_actions,
            frame_id_end=frame_id_end,
            info=info,
        )

        final_action_id = ACTION_NAME_TO_ID[final_action]
        behavior_probs[final_action_id] += prob[proposed_action_id]
        action_mapping[proposed_action_id] = final_action_id

        if not np.isclose(behavior_probs.sum(), 1.0, atol=1e-5,):
            raise ValueError("Behavior probabilities do not sum to 1.0")

    return behavior_probs, action_mapping