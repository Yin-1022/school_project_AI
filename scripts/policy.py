import random
import constant as const

def init_state():
    state = {
        "fsm_state": "Patrol",
        "hold_until_frame": -1,
        "cooldowns" : {"EvadeBack":0,"SearchTurn":0,"PatrolStep":0},
        "last_cmd": "Hold", 
        "last_cmd_at_frame": -1,
        "rt_queue": []
    }
    return state

def is_ready(state, key, now_frame):
    return now_frame >= state["cooldowns"].get(key, 0)

def arm_cooldown(state, key, fire_frame, cd_frames):
    state["cooldowns"][key] = fire_frame + cd_frames

def step(state, *, pred_name, conf, visible, phase, search_hint, frame_id_end):
    fsm_state = state["fsm_state"]
    hold_until_frame = state["hold_until_frame"]
    last_cmd = state["last_cmd"]
    rt_queue = state["rt_queue"]

    cmd = "Hold"
    params = {}
    new_cmd_selected = False

    if visible==1 and pred_name=="attack":
            fsm_state = "Evade"
            new_cmd_selected = True
            if is_ready(state, "EvadeBack", frame_id_end) and conf>=const.TAU_CMD:
                cmd = "EvadeBack"
            else:
                cmd = "Retreat"

    if not new_cmd_selected and frame_id_end < hold_until_frame:
        cmd = last_cmd
        state["fsm_state"] = fsm_state
        return cmd, state, {}, None
    else:
        # Normal FSM logic here 
        if visible==0:
            if phase=="patrol":
                fsm_state = "Patrol"
                if is_ready(state, "PatrolStep", frame_id_end):
                    cmd = "PatrolStep"
                    if search_hint in ["left","right"]:
                        params={"direction": search_hint}
                    else:
                        params={"direction": random.choice(['left','right'])}
                else:
                    cmd = "Hold"
            elif phase=="reacq":
                fsm_state = "Search"
                if is_ready(state, "SearchTurn", frame_id_end):
                    cmd = "SearchTurn"
                    if search_hint == "left":
                        params={"direction": "left"}
                    elif search_hint == "right":
                        params={"direction": "right"}
                    else:
                        params={"direction": random.choice(['left','right'])}
                else:
                    if is_ready(state, "PatrolStep", frame_id_end):
                        fsm_state = "Patrol"
                        cmd = "PatrolStep"
                        if search_hint in ["left","right"]:
                            params={"direction": search_hint}
                        else:
                            params={"direction": random.choice(['left','right'])}
                    else:
                        cmd = "Hold"  
        elif visible==1 and pred_name in {"move", "idle", "roll"}:
            fsm_state = "Chase"
            # rand_outcome = random.random() < 0.2 
            # if rand_outcome:
            n = (frame_id_end // const.CLIP_STRIDE)
            if n % 5 == 0:
                if search_hint == "left":
                    cmd = "StrafeLeft"
                elif search_hint == "right":
                    cmd = "StrafeRight"
                else:
                    cmd = "Advance"
            else:
                cmd = "Advance"
        else:
            fsm_state = "Patrol"
            cmd = "Hold"
        
    if new_cmd_selected or cmd != last_cmd:
        fire_frame = frame_id_end + const.RT_FRAMES
        state["hold_until_frame"] = fire_frame + const.MIN_HOLD_FRAMES
        state["last_cmd"] = cmd
        state["last_cmd_at_frame"] = frame_id_end

        if cmd == "EvadeBack":
            arm_cooldown(state, "EvadeBack", fire_frame, const.CD_EVADE)
        elif cmd == "SearchTurn":
            arm_cooldown(state, "SearchTurn", fire_frame, const.CD_TURN)
        elif cmd == "PatrolStep":
            arm_cooldown(state, "PatrolStep", fire_frame, const.CD_PATROL)
    else:
        fire_frame = None  # 沒有新指令，不產生新的 fire/hold

    # Update state
    state["fsm_state"] = fsm_state
    state["rt_queue"] = rt_queue

    return cmd, state, params, fire_frame