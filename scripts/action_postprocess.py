from constant import SAME_ACTION_REFIRE_FRAMES, RT_FRAMES, MIN_HOLD_FRAMES, CD_EVADE, CD_TURN, CD_PATROL
from policy import is_ready, arm_cooldown

def apply_action_with_state(pol_state, proposed_action, topk_actions, frame_id_end, info):
    hold_until_frame = pol_state.get("hold_until_frame", -1)
    last_action = pol_state.get("last_action", "Hold")
    last_proposed_action = pol_state.get("last_proposed_action", None)
    same_action_streak = pol_state.get("same_action_streak", 0)
    hold_streak = pol_state.get("hold_streak", 0)
    last_non_hold_action = pol_state.get("last_non_hold_action", "Hold")

    if frame_id_end < hold_until_frame:
        if last_action != "Hold":
            return last_action, pol_state, None
    
    if proposed_action == last_proposed_action:
        same_action_streak += 1
    else:
        same_action_streak = 1

    pol_state["last_proposed_action"] = proposed_action
    pol_state["same_action_streak"] = same_action_streak

    action = proposed_action
    visible = info.get("visible", 0)
    phase = info.get("phase", "patrol")
    search_hint = info.get("search_hint", None)
    pred_name = info.get("pred_name", "none")
    lost_visible_streak = info.get("lost_visible_streak", 0)

    if proposed_action == "Hold":
        hold_streak += 1
    else:
        hold_streak = 0

    if visible == 0 and phase == "track":
        last_chase_action = last_action if last_action in {"Advance", "StrafeLeft", "StrafeRight"} else None

        # grace window：短暫延續上一個追擊動作，但不要採納新的 Advance proposal
        if last_chase_action is not None and lost_visible_streak <= 2:
            action = last_chase_action
        else:
            if search_hint == "left":
                action = "SearchTurnLeft"
            elif search_hint == "right":
                action = "SearchTurnRight"
            else:
                action = "Hold"

    if visible == 1 and phase == "track":
        if action in {"SearchTurnLeft", "SearchTurnRight", "PatrolStepLeft", "PatrolStepRight"}:
            if search_hint == "left":
                action = "StrafeLeft"
            elif search_hint == "right":
                action = "StrafeRight"
            else:
                action = "Advance"
        if action in {"EvadeBack", "Retreat"} and pred_name not in {"attack", "roll"}:
            fallback = None

            # 先優先找橫移
            for cand in topk_actions:
                if cand in {"StrafeLeft", "StrafeRight"}:
                    fallback = cand
                    break

            # 再找 Advance
            if fallback is None:
                for cand in topk_actions:
                    if cand == "Advance":
                        fallback = cand
                        break

            # 最後真的沒得選才保留原 action
            if fallback is not None:
                action = fallback

    # track 階段不允許 SearchTurn
    if (
        visible == 1
        and phase == "track"
        and action in {"SearchTurnLeft", "SearchTurnRight"}
    ):
        if search_hint == "left":
            action = "StrafeLeft"
        elif search_hint == "right":
            action = "StrafeRight"
        else:
            action = "Advance"

    if visible == 1 and phase == "track":
        if action == "Retreat" and same_action_streak > 2:
            fallback = None

            # 先優先找橫移
            for cand in topk_actions:
                if cand in {"StrafeLeft", "StrafeRight"} and cand != action:
                    fallback = cand
                    break

            # 再找 Advance
            if fallback is None:
                for cand in topk_actions:
                    if cand == "Advance":
                        fallback = cand
                        break

            if fallback is not None:
                action = fallback
                pol_state["same_action_streak"] = 1
                pol_state["last_proposed_action"] = action
    
    if visible == 1 and phase == "track" and search_hint == "center":
        if action == "Retreat":
            fallback = None

            for cand in topk_actions:
                if cand == "Advance":
                    fallback = cand
                    break

            if fallback is None:
                for cand in topk_actions:
                    if cand in {"StrafeLeft", "StrafeRight"}:
                        fallback = cand
                        break

            if fallback is not None:
                action = fallback

    if phase == "reacq":
        if action == "Advance":
            if search_hint == "left":
                action = "SearchTurnLeft"
            elif search_hint == "right":
                action = "SearchTurnRight"
            else:
                action = "Hold"

    if phase == "patrol":
        if action in {"Advance", "SearchTurnLeft", "SearchTurnRight"}:
            if search_hint == "right":
                action = "PatrolStepRight"
            else:
                action = "PatrolStepLeft"
        elif action == "Hold" and hold_streak > 2:
            if search_hint == "left":
                action = "PatrolStepLeft"

            elif search_hint == "right":
                action = "PatrolStepRight"

            else:
                if last_non_hold_action == "PatrolStepLeft":
                    action = "PatrolStepRight"
                else:
                    action = "PatrolStepLeft"

            hold_streak = 0

    if action in {"StrafeRight", "StrafeLeft"} and same_action_streak > 3:
        fallback = None
        for cand in topk_actions:
            if cand != action:
                fallback = cand
                break
        if fallback is not None:
            action = fallback
            pol_state["same_action_streak"] = 1
            pol_state["last_proposed_action"] = action

    if action == "EvadeBack" and not is_ready(pol_state, "EvadeBack", frame_id_end):
        action = "Retreat"
    
    if action in {"SearchTurnLeft", "SearchTurnRight"} and not is_ready(pol_state, "SearchTurn", frame_id_end):
        action = last_action if frame_id_end < hold_until_frame else "Hold"

    if action in {"PatrolStepLeft", "PatrolStepRight"} and not is_ready(pol_state, "PatrolStep", frame_id_end):
        action = last_action if frame_id_end < hold_until_frame else "Hold"

    if action != "Hold":
        last_non_hold_action = action

    if action == "Hold" and hold_streak > 2:
        fallback = None
        for cand in topk_actions:
            if cand != "Hold" and cand != last_non_hold_action:
                fallback = cand
                break
        
        if fallback is None:
            for cand in topk_actions:
                if cand != "Hold":
                    fallback = cand
                    break

        if fallback is not None:
            action = fallback
            hold_streak = 0
            last_non_hold_action = action
    
    if action == "Hold":
        pol_state["hold_streak"] = hold_streak
        pol_state["last_non_hold_action"] = last_non_hold_action

        if hold_streak <= 2:
            visible = info.get("visible", 0)
            phase = info.get("phase", "patrol")

            # 1) track：可以沿用 chase 類動作
            if visible == 1 and phase == "track":
                continued_action = last_action
                if continued_action in {
                    "Advance",
                    "StrafeLeft",
                    "StrafeRight",
                }:
                    last_fire_at = pol_state.get("last_action_at_frame", -1)
                    if (frame_id_end - last_fire_at) >= SAME_ACTION_REFIRE_FRAMES:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action_at_frame"] = frame_id_end
                        return continued_action, pol_state, fire_frame
                    return continued_action, pol_state, None

            # 2) reacq：優先 SearchTurn，其次 PatrolStep
            if phase == "reacq":
                for cand in topk_actions:
                    if cand in {"SearchTurnLeft", "SearchTurnRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end  + MIN_HOLD_FRAMES
                        return cand, pol_state, fire_frame

                for cand in topk_actions:
                    if cand in {"PatrolStepLeft", "PatrolStepRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end  + MIN_HOLD_FRAMES
                        return cand, pol_state, fire_frame

                return "Hold", pol_state, None

            # 3) patrol：優先 PatrolStep，不要沿用 Advance
            if phase == "patrol":
                for cand in topk_actions:
                    if cand in {"PatrolStepLeft", "PatrolStepRight"}:
                        fire_frame = frame_id_end + RT_FRAMES
                        pol_state["last_action"] = cand
                        pol_state["last_action_at_frame"] = frame_id_end
                        pol_state["hold_until_frame"] = frame_id_end + MIN_HOLD_FRAMES
                        return cand, pol_state, fire_frame

                return "Hold", pol_state, None

    if action == last_action:
        last_fire_at = pol_state.get("last_action_at_frame", -1)
        if (frame_id_end - last_fire_at) < SAME_ACTION_REFIRE_FRAMES:
            pol_state["hold_streak"] = hold_streak
            pol_state["last_non_hold_action"] = last_non_hold_action
            return action, pol_state, None

    fire_frame = frame_id_end + RT_FRAMES
    if action != "Hold":
        pol_state["last_action"] = action
        pol_state["last_action_at_frame"] = frame_id_end
        pol_state["hold_until_frame"] = fire_frame + MIN_HOLD_FRAMES
    else:
        pol_state["hold_until_frame"] = frame_id_end + MIN_HOLD_FRAMES
    pol_state["hold_streak"] = hold_streak
    pol_state["last_non_hold_action"] = last_non_hold_action

    if action == "EvadeBack":
        arm_cooldown(pol_state, "EvadeBack", fire_frame, CD_EVADE)
    elif action == "SearchTurnLeft" or action == "SearchTurnRight":
        arm_cooldown(pol_state, "SearchTurn", fire_frame, CD_TURN)
    elif action == "PatrolStepLeft" or action == "PatrolStepRight":
        arm_cooldown(pol_state, "PatrolStep", fire_frame, CD_PATROL)

    return action, pol_state, fire_frame