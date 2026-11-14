import cv2
from pathlib import Path
import collections
from action_mask import build_action_mask
import torch
import numpy as np
from actor_config import ActorConfig, parse_actor_config
from presence_data import save_presence_sample
from visibility import update as vis_update
from policy import init_state as policy_init, step as rule_policy_step
from stream_io import reset_ue_episode_state, send_action, receive_from_ue, tcp_frame_stream, get_osc_client
from observation_builder import build_frame_tensor, build_extra_tensor, ACTION_NAME_TO_ID
from policy_inference import (
    load_actor_critic_model, load_model, 
    infer_action, infer_actor_critic_action,
    load_action_cls_model, infer_player_state, 
    reload_AC_if_newer,
    CLASS_TO_ID
)
from presence_inference import (
    load_presence_model,
    infer_player_presence,
)
from action_postprocess import apply_action_with_state
import threading
import time
from time import monotonic
from constant import (
    ACTION_ID_TO_NAME,
    POLICY_MODE,
    ACTION_MASK_MODE,
    PLAYER_MAX_HP,
    BOSS_MAX_HP,
    LEARNER_CHECKPOINT_PATH,
    ACTOR_CHECKPOINT_PATH
)
from behavior_policy import compute_behavior_probs
from episode_metrics import EpisodeEventAudit, PendingCombatEvents, format_summary
from ground_truth_audit import GroundTruthAudit, format_gt_summary
from transition_tracker import TransitionTracker
from runtime.controller import RuntimeController
from runtime.types import ExecutionResult, RuntimeAction, ActionSource, ExecutionEndReason
from runtime.adapters.impala_adapter import to_runtime_action, to_osc_payload

# RAW_DIR = Path("data/raw_videos")
# video_path = RAW_DIR / "raw_video_4_t.mp4"
WEIGHTS_PATH = Path("data/meta/best_teacher_policy.pt")
ACTION_CLS_WEIGHTS_PATH = Path("data/meta/best_action_cls.pt")
PRESENCE_WEIGHTS_PATH = Path("data/meta/best_presence_avgmax_balanced_hardP.pt")
CLIP_FRAMES     = 8          # 每個 clip 的影格數
CLIP_STRIDE     = 4          # 滑窗步長
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
SEQ = 0
UE_EVENT_STATE = {
    "att1_active": False,
    "att1_start_pulse": False,
    "att1_end_pulse": False,

    "att2_active": False,
    "att2_start_pulse": False,
    "att2_end_pulse": False,

    "att1_deadline": None,
    "att2_deadline": None,

    "episode_done_flag": False,
    "episode_start_pulse": False,
    "episode_result": 0,

    "act_active": False,
    "act_start_pulse": False,
    "act_end_pulse": False,
    "cantmove_pulse": False,
    "notseeplayer_pulse": False,
}
UE_EVENT_LOCK = threading.Lock()
PRESENCE_RECORD_MODE = True
PRESENCE_VIDEO_DIR = Path("data/presence_videos")

def main(config: ActorConfig):
    tracker = None
    video_writer = None
    reconnect_pending = False
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = None
        loaded_step = 0
        if POLICY_MODE == "bc":
            model = load_model(str(WEIGHTS_PATH), device=device)
        elif POLICY_MODE == "impala":
            model, loaded_step = load_actor_critic_model(str(ACTOR_CHECKPOINT_PATH), device=device)
            print(f"Loaded IMPALA actor step={loaded_step}")
            print(f"[Actor {config.actor_id}]")
            print(f"frame_port={config.frame_port}")
            print(f"action_port={config.action_port}")
            print(f"event_port={config.event_port}")

        action_client = get_osc_client(
            host=config.ue_host,
            action_port=config.action_port,
        )
        event_audit = EpisodeEventAudit(player_max_hp=PLAYER_MAX_HP, boss_max_hp=BOSS_MAX_HP)
        pending_combat = PendingCombatEvents()
        ground_truth_audit = GroundTruthAudit()
        receive_from_ue(UE_EVENT_LOCK, UE_EVENT_STATE, event_audit, pending_combat, ground_truth_audit, event_port=config.event_port)
        action_cls_model = load_action_cls_model(str(ACTION_CLS_WEIGHTS_PATH), device=device)

        presence_model = load_presence_model(
            str(PRESENCE_WEIGHTS_PATH),
            device=device,
        )
        presence_state = init_presence_state()

        vis_state = None                    
        pol_state = policy_init() 
        runtime = RuntimeController()

        frame_ring_buffer = collections.deque(maxlen=CLIP_FRAMES)
        pushed_frames = 0
        recv_frames = 0
        sample_every = 1
        decision_count = 0
        terminal_handled = False
        force_decision = False
        pending_rejection_result = None
        global SEQ

        print("ACTION_MASK_MODE: ", ACTION_MASK_MODE)

        for frame in tcp_frame_stream(host=config.frame_host, frame_port=config.frame_port, img_w=192, img_h=192, img_c=3, debug_show=False):
            if frame is None:
                episode_done_now = False
                episode_result_now = 0

                runtime.reset()
                pending_rejection_result = None

                # 給 OSC callback 一點時間把 terminal result 寫進 shared state
                for _ in range(10):   # 最多等 10 * 0.02 = 0.2 秒
                    with UE_EVENT_LOCK:
                        episode_done_now = UE_EVENT_STATE["episode_done_flag"]
                        episode_result_now = UE_EVENT_STATE["episode_result"]
                    if episode_done_now:
                        break
                    time.sleep(0.02)

                if episode_done_now and not terminal_handled:
                    print("[UE event] episode done (disconnect terminal append)")

                    terminal_combat_state = capture_terminal_combat_state(event_audit, pending_combat)

                    tracker.record_terminal_combat(terminal_combat_state)

                    event_audit_snapshot = {
                        "player_hits_on_boss_raw":terminal_combat_state["raw_player_hits_on_boss"],
                        "boss_hits_on_player_raw":terminal_combat_state["raw_boss_hits_on_player"],
                        "unattributed_boss_hits_on_player":event_audit.unattributed_boss_hits_on_player,
                        "unattributed_player_hits_on_boss":event_audit.unattributed_player_hits_on_boss,
                        "unattributed_boss_damage_to_player":event_audit.unattributed_boss_damage_to_player,
                        "unattributed_player_damage_to_boss":event_audit.unattributed_player_damage_to_boss,
                        "boss_damage_to_player_raw":terminal_combat_state["boss_damage_to_player_raw"],
                        "player_damage_to_boss_raw":terminal_combat_state["player_damage_to_boss_raw"],
                        "final_player_health":terminal_combat_state["final_player_health"],
                        "final_boss_health":terminal_combat_state["final_boss_health"],
                        "player_max_hp": event_audit.player_max_hp,
                        "boss_max_hp": event_audit.boss_max_hp
                    }

                    finalize_episode(
                        tracker=tracker,
                        event_audit=event_audit_snapshot,
                        ground_truth_audit=ground_truth_audit,
                        episode_result=episode_result_now,
                        policy_step=loaded_step,
                    )

                    terminal_handled = True
                    tracker.clear_pending()
                    with UE_EVENT_LOCK:
                        UE_EVENT_STATE["episode_done_flag"] = False
                        UE_EVENT_STATE["episode_result"] = 0
                else:
                    tracker.drop_buffer_on_disconnect()

                    if not terminal_handled:
                        tracker.reset_episode_state()

                        with UE_EVENT_LOCK:
                            event_audit.reset()
                            pending_combat.reset()
                            ground_truth_audit.reset()

                print("[stream] disconnected, waiting for UE reconnect")
                tracker.clear_pending()
                reconnect_pending = True
                continue

            if reconnect_pending:
                print("[stream] UE reconnected, resetting runtime state")

                presence_state = init_presence_state()
                vis_state = None
                pol_state = policy_init()

                frame_ring_buffer.clear()
                pushed_frames = 0
                recv_frames = 0

                runtime.reset()
                pending_rejection_result = None

                force_decision = False

                terminal_handled = False

                tracker.reset_episode_state()

                with UE_EVENT_LOCK:
                    event_audit.reset()
                    pending_combat.reset()
                    ground_truth_audit.reset()

                    UE_EVENT_STATE["episode_done_flag"] = False
                    UE_EVENT_STATE["episode_result"] = 0

                reset_ue_episode_state(
                    UE_EVENT_LOCK=UE_EVENT_LOCK,
                    UE_EVENT_STATE=UE_EVENT_STATE,
                )

                reconnect_pending = False

            with UE_EVENT_LOCK:
                episode_started = UE_EVENT_STATE["episode_start_pulse"]
                episode_done = UE_EVENT_STATE["episode_done_flag"]
                episode_result = UE_EVENT_STATE["episode_result"]

                if episode_started:
                    UE_EVENT_STATE["episode_start_pulse"] = False

            if episode_done and episode_result and not terminal_handled:

                terminal_combat_state = capture_terminal_combat_state(event_audit, pending_combat)

                tracker.record_terminal_combat(terminal_combat_state)

                event_audit_snapshot = {
                    "player_hits_on_boss_raw":terminal_combat_state["raw_player_hits_on_boss"],
                    "boss_hits_on_player_raw":terminal_combat_state["raw_boss_hits_on_player"],
                    "unattributed_boss_hits_on_player":event_audit.unattributed_boss_hits_on_player,
                    "unattributed_player_hits_on_boss":event_audit.unattributed_player_hits_on_boss,
                    "unattributed_boss_damage_to_player":event_audit.unattributed_boss_damage_to_player,
                    "unattributed_player_damage_to_boss":event_audit.unattributed_player_damage_to_boss,
                    "boss_damage_to_player_raw":terminal_combat_state["boss_damage_to_player_raw"],
                    "player_damage_to_boss_raw":terminal_combat_state["player_damage_to_boss_raw"],
                    "final_player_health":terminal_combat_state["final_player_health"],
                    "final_boss_health":terminal_combat_state["final_boss_health"],
                    "player_max_hp": event_audit.player_max_hp,
                    "boss_max_hp": event_audit.boss_max_hp
                }

                finalize_episode(
                    tracker=tracker,
                    event_audit=event_audit_snapshot,
                    ground_truth_audit=ground_truth_audit,
                    episode_result=episode_result,
                    policy_step=loaded_step,
                )
                terminal_handled = True

                with UE_EVENT_LOCK:
                    UE_EVENT_STATE["episode_done_flag"] = False
                    UE_EVENT_STATE["episode_result"] = 0

            if episode_started:
                print("[episode] New episode started, resetting runtime state")
                presence_state = init_presence_state()
                vis_state = None
                pol_state = policy_init()

                frame_ring_buffer.clear()

                pushed_frames = 0
                recv_frames = 0
                pending_rejection_result = None
                runtime.reset()
                force_decision = False
                terminal_handled = False
                tracker.reset_episode_state()

                with UE_EVENT_LOCK:
                    event_audit.reset()
                    pending_combat.reset()
                    ground_truth_audit.reset()

                reset_ue_episode_state(UE_EVENT_LOCK=UE_EVENT_LOCK, UE_EVENT_STATE=UE_EVENT_STATE)
            elif terminal_handled:
                continue

            recv_frames += 1

            if recv_frames % sample_every != 0:
                continue

            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            
            if PRESENCE_RECORD_MODE:
                if video_writer is None:
                    PRESENCE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

                    timestamp = int(time.time())
                    video_path = (
                        PRESENCE_VIDEO_DIR
                        / f"presence_raw_{config.actor_id}_{timestamp}.mp4"
                    )

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        TARGET_FPS,
                        FRAME_SIZE,
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Failed to open video writer: {video_path}"
                        )

                    print(f"[presence-record] started: {video_path}")

                video_writer.write(frame)
            
            frame_ring_buffer.append(frame)
            pushed_frames += 1

            # =====================================
            # Lightweight Execution Event Polling
            # 每張 frame 都檢查，不等 CLIP_STRIDE
            # =====================================
            with UE_EVENT_LOCK:
                ue_act_start_now = UE_EVENT_STATE["act_start_pulse"]
                ue_act_end_now = UE_EVENT_STATE["act_end_pulse"]
                cantmove_now = UE_EVENT_STATE["cantmove_pulse"]
                notseeplayer_now = UE_EVENT_STATE["notseeplayer_pulse"]

                UE_EVENT_STATE["act_start_pulse"] = False
                UE_EVENT_STATE["act_end_pulse"] = False
                UE_EVENT_STATE["cantmove_pulse"] = False
                UE_EVENT_STATE["notseeplayer_pulse"] = False

            expired_attacks = runtime.attack.check_timeouts(now=monotonic())
            current_action = runtime.execution.state.current_action
            current_name = current_action.name if current_action else None

            for attack in expired_attacks:
                print(f"[attack watchdog] {attack} timeout -> auto unfreeze")

            # =====================================
            # Shared Runtime Execution Lifecycle
            # =====================================
            now = monotonic()

            # Action Start
            if ue_act_start_now:
                runtime.on_action_started()

            # Strafe Rejection
            rejected_this_poll = False

            if notseeplayer_now:
                rejected_result = runtime.on_notseeplayer()

                if (
                    rejected_result is not None
                    and rejected_result.reason == ExecutionEndReason.REJECTED
                ):
                    pending_rejection_result = rejected_result
                    rejected_this_poll = True
                    force_decision = True

                    print(
                        f"[execution] {rejected_result.action.name} "
                        "rejected by /notseeplayer"
                    )

            if cantmove_now and not rejected_this_poll:
                runtime.on_cantmove()
                force_decision = True

            # Action End
            if ue_act_end_now and not rejected_this_poll:
                result = runtime.on_action_ended()

                if result is not None:
                    finished_action = result.action.name

                    print(
                        f"[execution] {finished_action} finished by /actend"
                    )

                    # 保留 Ground Truth Audit 的原始 Policy Name
                    with UE_EVENT_LOCK:
                        ground_truth_audit.finish_action(result.action)

                    force_decision = True

            # Execution Timeout
            timeout_result = runtime.tick(now=now)

            if timeout_result is not None:
                if timeout_result.reason in {
                    ExecutionEndReason.START_TIMEOUT,
                    ExecutionEndReason.END_TIMEOUT,
                }:
                    print(
                        f"[execution watchdog] "
                        f"{timeout_result.action.name} "
                        f"{timeout_result.reason.value}"
                    )

                    with UE_EVENT_LOCK:
                        ground_truth_audit.abort_action(
                            timeout_result.reason.value
                        )

                    force_decision = True

            if len(frame_ring_buffer) < CLIP_FRAMES:
                continue

            if (not force_decision) and (pushed_frames % CLIP_STRIDE != 0):
                continue
            force_decision = False

            frame_id_end = pushed_frames - 1
            frames = build_frame_tensor(frame_ring_buffer)

            presence_prob = infer_player_presence(
                frames=frames,
                model=presence_model,
            )

            presence_raw = int(presence_prob >= 0.60)

            presence_stable = update_presence_state(
                state=presence_state,
                probability=presence_prob,
                enter_threshold=0.60,
                exit_threshold=0.45,
                enter_required=2,
                exit_required=3,
            )

            pred_out = infer_player_state(frames, action_cls_model)
            raw_pred_name = pred_out["pred_name"]
            pred_conf = pred_out["conf"]
            pred_topk_ids = pred_out["topk_ids"][0]
            pred_topk_probs = pred_out["topk_probs"][0]
            pred_topk_names = [pred_out["pred_name"]]  # 先下面再補更完整 mapping
            none_id = CLASS_TO_ID["none"]
            none_prob = pred_out["probs"][0, none_id].item()

            if presence_stable == 1:
                pred_name = raw_pred_name
            else:
                pred_name = "none"

            info, vis_state = vis_update(
                vis_state,
                frames,
                pred_name=pred_name,
                visible=presence_stable,
                frame_id_end=frame_id_end,
            )

            runtime.on_visibility_updated(bool(info["visible"]))

            extra_tensor = build_extra_tensor(info, pol_state, frame_id_end)
            now = monotonic()

            with UE_EVENT_LOCK:
                ue_att1_active = UE_EVENT_STATE["att1_active"]
                ue_att1_start = UE_EVENT_STATE["att1_start_pulse"]
                ue_att1_end = UE_EVENT_STATE["att1_end_pulse"]

                ue_att2_active = UE_EVENT_STATE["att2_active"]
                ue_att2_start = UE_EVENT_STATE["att2_start_pulse"]
                ue_att2_end = UE_EVENT_STATE["att2_end_pulse"]

                ue_boss_hit_count = pending_combat.boss_hit_count
                ue_player_hit_count = pending_combat.player_hit_count
                ue_boss_damage = pending_combat.boss_damage
                ue_player_damage = pending_combat.player_damage
                ue_episode_done = UE_EVENT_STATE["episode_done_flag"]
                ue_episode_result = UE_EVENT_STATE["episode_result"]

                # pulse 讀完就清掉
                UE_EVENT_STATE["att1_start_pulse"] = False
                UE_EVENT_STATE["att1_end_pulse"] = False
                UE_EVENT_STATE["att2_start_pulse"] = False
                UE_EVENT_STATE["att2_end_pulse"] = False
                pending_combat.reset()

            for attack_type, started, ended, active in [
                ("att1", ue_att1_start, ue_att1_end, ue_att1_active),
                ("att2", ue_att2_start, ue_att2_end, ue_att2_active),
            ]:
                if started and ended:
                    # 同一批次同時收到 start/end，依最後 active 狀態決定結果
                    if active:
                        runtime.on_attack_end(attack_type)
                        runtime.on_attack_start(attack_type)
                    else:
                        runtime.on_attack_start(attack_type)
                        runtime.on_attack_end(attack_type)
                elif started:
                    runtime.on_attack_start(attack_type)
                elif ended:
                    runtime.on_attack_end(attack_type)

            tracker.record_frame_events(
                boss_hit_count=ue_boss_hit_count,
                player_hit_count=ue_player_hit_count,
                boss_damage=ue_boss_damage,
                player_damage=ue_player_damage,
                att1_start=ue_att1_start,
                att1_end=ue_att1_end,
                att2_start=ue_att2_start,
                att2_end=ue_att2_end,
                episode_done=ue_episode_done,
                episode_result=ue_episode_result,
            )

            if pending_rejection_result is not None:

                rejected_result = pending_rejection_result
                pending_rejection_result = None

                print(
                    f"[execution fallback] {rejected_result.action.name} "
                    "rejected -> Advance"
                )

                # 這筆不能當成「Strafe 被成功執行」拿去訓練
                tracker.discard_rejected_policy()

                # 改由 runtime fallback 執行 Advance
                with UE_EVENT_LOCK:
                    ground_truth_audit.abort_action(
                        "notseeplayer_rejected"
                    )
            
            if ue_att1_start:
                print("[UE event] boss normal attack start")
            if ue_att1_end:
                print("[UE event] boss normal attack end")
            if ue_att2_start:
                print("[UE event] boss skill attack1 start")
            if ue_att2_end:
                print("[UE event] boss skill attack1 end")
            if ue_boss_hit_count>0:
                print(f"[UE event] boss hit x {ue_boss_hit_count}")
                runtime.on_boss_hit(visible=bool(info["visible"]))
            if ue_player_hit_count>0:
                print(f"[UE event] player hit x {ue_player_hit_count}")
            if ue_episode_done and not terminal_handled:
                print("[UE event] episode done")

                terminal_combat_state = capture_terminal_combat_state(event_audit, pending_combat)

                tracker.record_terminal_combat(terminal_combat_state)

                event_audit_snapshot = {
                    "player_hits_on_boss_raw":terminal_combat_state["raw_player_hits_on_boss"],
                    "boss_hits_on_player_raw":terminal_combat_state["raw_boss_hits_on_player"],
                    "unattributed_boss_hits_on_player":event_audit.unattributed_boss_hits_on_player,
                    "unattributed_player_hits_on_boss":event_audit.unattributed_player_hits_on_boss,
                    "unattributed_boss_damage_to_player":event_audit.unattributed_boss_damage_to_player,
                    "unattributed_player_damage_to_boss":event_audit.unattributed_player_damage_to_boss,
                    "boss_damage_to_player_raw":terminal_combat_state["boss_damage_to_player_raw"],
                    "player_damage_to_boss_raw":terminal_combat_state["player_damage_to_boss_raw"],
                    "final_player_health":terminal_combat_state["final_player_health"],
                    "final_boss_health":terminal_combat_state["final_boss_health"],
                    "player_max_hp": event_audit.player_max_hp,
                    "boss_max_hp": event_audit.boss_max_hp
                }

                finalize_episode(
                    tracker=tracker,
                    event_audit=event_audit_snapshot,
                    ground_truth_audit=ground_truth_audit,
                    episode_result=ue_episode_result,
                    policy_step=loaded_step,
                )
                terminal_handled = True

                with UE_EVENT_LOCK:
                    UE_EVENT_STATE["episode_done_flag"] = False
                    UE_EVENT_STATE["episode_result"] = 0

                continue

            current_action = runtime.execution.state.current_action

            if not runtime.can_send():
                if runtime.attack.is_attacking():
                    print(
                        f"[decision freeze] attack_active=1 "
                        f"at t={frame_id_end:05d}, skip new inference"
                    )
                continue

            tracker.rotate_at_decision_boundary()
 
            # Shared Runtime Pending Actions
            runtime_action = runtime.next_action()

            if runtime_action is not None:
                source_map = {
                    ActionSource.RUNTIME_RECOVERY: "recovery",
                    ActionSource.RUNTIME_FALLBACK: "runtime_fallback",
                    ActionSource.RUNTIME_HIT_REACTION: "hit_reaction",
                }

                with UE_EVENT_LOCK:
                    ground_truth_audit.begin_action(
                        action=runtime_action.name,
                        source=source_map[runtime_action.source],
                        expected_runtime_action=runtime_action,
                        pred_visible=int(info["visible"]),
                        presence_prob=float(presence_prob),
                        frame_id_end=frame_id_end,
                        policy_step=loaded_step,
                    )

                send_runtime_action(
                    runtime_action,
                    action_client,
                    runtime,
                )

                continue

            action_mask = build_action_mask(pol_state, frame_id_end, info, mode=ACTION_MASK_MODE)

            if POLICY_MODE == "bc":
                policy_out = infer_action(frames, extra_tensor, model, sample=True, action_mask=action_mask)

            elif POLICY_MODE == "impala":
                policy_out = infer_actor_critic_action(frames, extra_tensor, model, sample=True, action_mask=action_mask)

            elif POLICY_MODE == "rule":
                rule_pred_name, rule_conf = derive_rule_pred_name(info)

                proposed_action, pol_state, _, fire_frame = rule_policy_step(
                    pol_state,
                    pred_name=rule_pred_name,
                    conf=rule_conf,
                    visible=info["visible"],
                    phase=info["phase"],
                    search_hint=info["search_hint"],
                    frame_id_end=frame_id_end,
                )

                action = proposed_action
                action_conf = 1.0
                topk_actions = [proposed_action]
                topk_confs = [1.0]

                # rollout_logger 仍需要 logits/probs，先放假資料
                num_actions = len(ACTION_ID_TO_NAME)
                logits_np = np.zeros((1, num_actions), dtype=np.float32)
                probs_np = np.zeros((1, num_actions), dtype=np.float32)
                action_id = ACTION_NAME_TO_ID[proposed_action]
                logits_np[0, action_id] = 1.0
                probs_np[0, action_id] = 1.0

                logits_for_log = torch.from_numpy(logits_np)
                probs_for_log = torch.from_numpy(probs_np)

            if POLICY_MODE in ["bc", "impala"]:
                proposed_action = policy_out["action_name"]
                action_conf = policy_out["conf"]
                topk_actions = [ACTION_ID_TO_NAME[id] for id in policy_out["topk_ids"][0]]
                topk_confs = policy_out["topk_probs"][0]
                logits_for_log = policy_out["logits"]
                probs_for_log = policy_out["probs"]

                policy_probs_np = (probs_for_log.squeeze(0).detach().cpu().numpy())

                behavior_probs, action_mapping = compute_behavior_probs(
                    prob=policy_probs_np,
                    pol_state=pol_state,
                    topk_actions=topk_actions,
                    frame_id_end=frame_id_end,
                    info=info,
                    action_mask=action_mask
                )

                action, pol_state, fire_frame = apply_action_with_state(
                    pol_state,
                    proposed_action=proposed_action,
                    topk_actions=topk_actions,
                    frame_id_end=frame_id_end,
                    info=info
                )

                proposed_action_id = ACTION_NAME_TO_ID[proposed_action]
                expected_final_action_id = action_mapping[proposed_action_id]
                actual_final_action_id = ACTION_NAME_TO_ID[action]

                if expected_final_action_id != actual_final_action_id:
                    raise RuntimeError(
                        "Simulated final action does not match "
                        "production final action"
                    )
            elif POLICY_MODE == "rule":
                behavior_probs = np.zeros(len(ACTION_ID_TO_NAME),dtype=np.float32)
                behavior_probs[ACTION_NAME_TO_ID[action]] = 1.0

            print(
                f"[t={frame_id_end:05d}] "
                f"presence={presence_prob:.2f} "
                f"raw={presence_raw} "
                f"stable={presence_stable} "
                f"streak=({presence_state['present_streak']},"
                f"{presence_state['absent_streak']}) "
                f"| raw_pred={raw_pred_name}({pred_conf:.2f}) "
                f"effective_pred={pred_name} "
                f"none_prob={none_prob:.2f} "
                f"visible={info['visible']} "
                f"phase={info['phase']} "
                f"hint={info['search_hint']} "
                f"motion={info['motion']:.4f} "
                f"→ proposed_action={proposed_action}({action_conf:.2f}) "
                f"final_action={action} "
                f"fire@{fire_frame} "
                f"hold_until={pol_state['hold_until_frame']} "
                f"topk={list(zip(topk_actions, topk_confs))}"
            )

            if fire_frame is None:
                continue

            SEQ += 1
            jsonMsg = {
                "type": "boss_action",
                "ts_frame": frame_id_end,
                "fire_frame": fire_frame,
                "hold_until": pol_state["hold_until_frame"],
                "action": action,
                "params": {},
                "meta": {
                    "conf": action_conf,
                    "phase": info["phase"],
                    "search_hint": info["search_hint"],
                },
                "seq": SEQ
            }

            selected_action = runtime.next_action(to_runtime_action(action))

            if (
                selected_action is None
                or selected_action.source != ActionSource.POLICY
            ):
                raise RuntimeError("Policy dispatch was unexpectedly preempted")

            with UE_EVENT_LOCK:
                ground_truth_audit.begin_action(
                    action=action,
                    source="policy",
                    expected_runtime_action=selected_action,
                    pred_visible=int(info["visible"]),
                    presence_prob=float(presence_prob),
                    frame_id_end=frame_id_end,
                    policy_step=loaded_step,
                )

            policy_cache = {
                "frames": frames,
                "extra": extra_tensor,
                "logits": logits_for_log,
                "probs": probs_for_log,
                "behavior_probs": behavior_probs,

                "proposed_action": proposed_action,
                "final_action": action,

                "info": info,
                "pol_state": dict(pol_state),

                "frame_id_end": frame_id_end,
                "fire_frame": fire_frame,

                "ue_att1_start": False,
                "ue_att1_end": False,
                "ue_att2_start": False,
                "ue_att2_end": False,
                "ue_boss_hit_count": 0,
                "ue_player_hit_count": 0,
                "ue_boss_damage": 0,
                "ue_player_damage": 0,
                "ue_episode_done": False,
                "ue_episode_result": 0,

                "action_mask": action_mask.copy(),
                "actor_policy_step": loaded_step,
            }

            tracker.record_policy_sent(policy_cache)

            send_runtime_action(
                selected_action,
                action_client,
                runtime,
            )

            print(
                f"[execution] sent {action}, waiting for /actstart"
            )

            if POLICY_MODE == "impala":
                decision_count+=1

                if decision_count %10 == 0:
                    loaded_step = reload_AC_if_newer(model, loaded_step, str(ACTOR_CHECKPOINT_PATH), device=device)

    finally:
        if video_writer is not None:
            video_writer.release()
            print("[presence-record] video saved")

        cv2.destroyAllWindows()

        if tracker is not None:
            tracker.flush()

def init_presence_state() -> dict:
    return {
        "visible": 0,
        "present_streak": 0,
        "absent_streak": 0,
    }


def update_presence_state(
    state: dict,
    probability: float,
    enter_threshold: float = 0.60,
    exit_threshold: float = 0.25,
    enter_required: int = 2,
    exit_required: int = 2,
) -> int:
    """
    raw present:
      probability >= 0.60

    raw absent:
      probability < 0.25

    0.25~0.60:
      模糊區域，暫時維持原 visible
    """

    if probability >= enter_threshold:
        state["present_streak"] += 1
        state["absent_streak"] = 0

    elif probability < exit_threshold:
        state["absent_streak"] += 1
        state["present_streak"] = 0

    else:
        # 落在 hysteresis 中間區，不切換狀態
        state["present_streak"] = 0
        state["absent_streak"] = 0

    if state["visible"] == 0:
        if state["present_streak"] >= enter_required:
            state["visible"] = 1
            state["present_streak"] = 0

    else:
        if state["absent_streak"] >= exit_required:
            state["visible"] = 0
            state["absent_streak"] = 0

    return state["visible"]

def derive_rule_pred_name(info):
    if info["visible"] == 0:
        return "none", 1.0

    if info["phase"] == "track":
        return "move", 0.8

    return "idle", 0.6

def finalize_episode(
    *,
    tracker: TransitionTracker,
    event_audit,
    ground_truth_audit,
    episode_result,
    policy_step,
):
    with UE_EVENT_LOCK:
        if ground_truth_audit.current is not None:
            ground_truth_audit.abort_action(
                "episode_terminal"
            )
    
    tracker.finalize_transitions(episode_result)
        

    summary = tracker.episode_metrics.finish_episode(
        episode_result=episode_result,
        event_audit=event_audit,
        policy_step=policy_step,
    )

    print(format_summary(summary))
    print(format_gt_summary(ground_truth_audit.build_summary()))

    tracker.flush()

def capture_terminal_combat_state(event_audit, pending_combat):

    with UE_EVENT_LOCK:
        pending_boss_hits = pending_combat.boss_hit_count
        pending_player_hits = pending_combat.player_hit_count
        pending_boss_damage = pending_combat.boss_damage
        pending_player_damage = pending_combat.player_damage

        # Consume pending hit events.
        pending_combat.reset()

        # Freeze raw OSC-event counts at the same moment.
        raw_player_hits_on_boss = event_audit.player_hits_on_boss_raw
        raw_boss_hits_on_player = event_audit.boss_hits_on_player_raw

        boss_damage_to_player_raw = event_audit.boss_damage_to_player_raw
        player_damage_to_boss_raw = event_audit.player_damage_to_boss_raw
        final_player_health = event_audit.final_player_health
        final_boss_health = event_audit.final_boss_health

    return {
        # Pending events still waiting for attribution
        "pending_boss_hits": pending_boss_hits,
        "pending_player_hits": pending_player_hits,
        "pending_boss_damage": pending_boss_damage,
        "pending_player_damage": pending_player_damage,

        # Raw UE events received during this episode
        "raw_player_hits_on_boss": raw_player_hits_on_boss,
        "raw_boss_hits_on_player": raw_boss_hits_on_player,

        "boss_damage_to_player_raw": boss_damage_to_player_raw,
        "player_damage_to_boss_raw": player_damage_to_boss_raw,
        "final_player_health": final_player_health,
        "final_boss_health": final_boss_health
    }

def send_runtime_action(action: RuntimeAction, action_client, runtime: RuntimeController) -> None:
    payload = to_osc_payload(action)
    send_action(payload, action_client=action_client)
    runtime.on_action_sent(action)

# def save_teacher_sample(frames, extra, action_name, frame_id_end):
#     out_dir = Path("data/teacher_samples")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     frames = frames.squeeze(0).detach().cpu().numpy()   # shape (C,T,H,W)
#     extra = extra.squeeze(0).detach().cpu().numpy()     # shape (24,)
#     if action_name not in ACTION_NAME_TO_ID:
#         raise ValueError(f"Unknown action_name: {action_name}")
#     action_id = ACTION_NAME_TO_ID[action_name]

#     timestamp = int(time.time() * 1000)
#     out_path = out_dir / f"sample_{timestamp}_{frame_id_end:06d}.npz"

#     np.savez(
#         out_path, 
#         frames=frames, 
#         extra=extra, 
#         action_id=np.int64(action_id),
#         action_name=action_name,
#         frame_id_end=np.int64(frame_id_end),
#     )
#     print(f"Saved teacher sample to {out_path}")
    
if __name__ == "__main__":
    config = parse_actor_config()
    main(config)