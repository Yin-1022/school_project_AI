import random
import constant as const

def init_state():
    state = {
        "fsm_state": "Patrol",
        "hold_until_frame": -1,
        "cooldowns" : {"EvadeBack":0,"SearchTurn":0,"PatrolStep":0},
        "last_action": "Hold", 
        "last_action_at_frame": -1,
        "rt_queue": [],
        "same_action_streak": 0,
        "last_proposed_action": None,
        "hold_streak": 0,
        "last_non_hold_action": "Hold",
    }
    return state

def is_ready(state, key, now_frame):
    return now_frame >= state["cooldowns"].get(key, 0)

def arm_cooldown(state, key, fire_frame, cd_frames):
    state["cooldowns"][key] = fire_frame + cd_frames

def step(state, *, pred_name, conf, visible, phase, search_hint, frame_id_end):
    fsm_state = state["fsm_state"]
    hold_until_frame = state["hold_until_frame"]
    last_action = state["last_action"]
    rt_queue = state["rt_queue"]

    action = "Hold"
    params = {}
    new_action_selected = False
    teacher_force_evasive = False

    if const.TEACHER_DATA_MODE:
        if visible == 1 and frame_id_end >= hold_until_frame:
            if random.random() < const.TEACHER_EVADE_PROB:
                teacher_force_evasive = True

    if visible == 1 and (pred_name == "attack" or teacher_force_evasive):
        fsm_state = "Evade"
        new_action_selected = True

        if is_ready(state, "EvadeBack", frame_id_end):
            if const.TEACHER_DATA_MODE and teacher_force_evasive:
                if random.random() < const.TEACHER_EVADEBACK_PROB:
                    action = "EvadeBack"
                else:
                    action = "Retreat"
            else:
                if conf >= const.TAU_ACTION:
                    action = "EvadeBack"
                else:
                    action = "Retreat"
        else:
            action = "Retreat"

    # if visible==1 and pred_name=="attack":
    #         fsm_state = "Evade"
    #         new_action_selected = True
    #         if is_ready(state, "EvadeBack", frame_id_end) and conf>=const.TAU_ACTION:
    #             action = "EvadeBack"
    #         else:
    #             action = "Retreat"

    if new_action_selected:
        pass
    elif frame_id_end < hold_until_frame:
        action = last_action
        state["fsm_state"] = fsm_state
        return action, state, {}, None
    else:
        # Normal FSM logic here 
        if visible==0:
            if phase=="patrol":
                fsm_state = "Patrol"
                if is_ready(state, "PatrolStep", frame_id_end):
                    if search_hint == "left":
                        action = "PatrolStepLeft"
                    elif search_hint == "right":
                        action = "PatrolStepRight"
                    else:
                        action = random.choice(["PatrolStepLeft", "PatrolStepRight"])
                else:
                    action = "Hold"
            elif phase=="reacq":
                fsm_state = "Search"
                if is_ready(state, "SearchTurn", frame_id_end):
                    if search_hint == "left":
                        action = "SearchTurnLeft"
                    elif search_hint == "right":
                        action = "SearchTurnRight"
                    else:
                        action = random.choice(["SearchTurnLeft", "SearchTurnRight"])
                else:
                    if is_ready(state, "PatrolStep", frame_id_end):
                        fsm_state = "Patrol"
                        if search_hint == "left":
                            action = "PatrolStepLeft"
                        elif search_hint == "right":
                            action = "PatrolStepRight"
                        else:
                            action = random.choice(["PatrolStepLeft", "PatrolStepRight"])
                    else:
                        action = "Hold"  
        elif visible==1 and pred_name in {"move", "idle", "roll"}:
            fsm_state = "Chase"
            # rand_outcome = random.random() < 0.2 
            # if rand_outcome:
            n = (frame_id_end // const.CLIP_STRIDE)
            # if n % 2 == 0:
            if search_hint == "left":
                action = "StrafeLeft"
            elif search_hint == "right":
                action = "StrafeRight"
            else:
                action = "Advance"
            # if n % 5 == 0:
            #     if search_hint == "left":
            #         action = "StrafeLeft"
            #     elif search_hint == "right":
            #         action = "StrafeRight"
            #     else:
            #         action = "Advance"
            # else:
            #     action = "Advance"
        else:
            fsm_state = "Patrol"
            action = "Hold"
        
    if new_action_selected or action != last_action:
        fire_frame = frame_id_end + const.RT_FRAMES
        state["hold_until_frame"] = fire_frame + const.MIN_HOLD_FRAMES
        state["last_action"] = action
        state["last_action_at_frame"] = frame_id_end

        if action == "EvadeBack":
            arm_cooldown(state, "EvadeBack", fire_frame, const.CD_EVADE)
        elif action in {"SearchTurnLeft", "SearchTurnRight"}:
            arm_cooldown(state, "SearchTurn", fire_frame, const.CD_TURN)
        elif action in {"PatrolStepLeft", "PatrolStepRight"}:
            arm_cooldown(state, "PatrolStep", fire_frame, const.CD_PATROL)
    else:
        fire_frame = None  # 沒有新指令，不產生新的 fire/hold

    # Update state
    state["fsm_state"] = fsm_state
    state["rt_queue"] = rt_queue

    return action, state, params, fire_frame