from dataclasses import dataclass, field
from collections import Counter
from time import monotonic

MOVEMENT_ACTIONS = {
    "Advance",
    "StrafeLeft",
    "StrafeRight",
}

SEARCH_ACTIONS = {
    "SearchTurnLeft",
    "SearchTurnRight",
    "PatrolStepLeft",
    "PatrolStepRight",

    # runtime actions
    "SearchTurn",
    "PatrolStep",
}

@dataclass
class GroundTruthActionRecord:
    action: str
    source: str

    pred_visible: int | None = None
    presence_prob: float | None = None
    frame_id_end: int | None = None
    policy_step: int | None = None

    gt_can_see_start: bool | None = None
    gt_distance_start: float | None = None

    gt_can_see_end: bool | None = None
    gt_distance_end: float | None = None

    completed: bool = False
    aborted_reason: str | None = None

@dataclass
class GroundTruthAudit:
    current: GroundTruthActionRecord | None = None
    records: list = field(default_factory=list)

    unmatched_status: int = 0
    unmatched_distance: int = 0
    action_mismatch: int = 0

    def begin_action(self, action: str, source: str, pred_visible: int | None = None, presence_prob: float | None = None, frame_id_end: int | None = None, policy_step: int | None = None):
        if self.current is not None:
            self.action_mismatch += 1
            self.current.completed = False
            self.current.aborted_reason = "begin_action called while another action is in progress"
            self.records.append(self.current)
        self.current = GroundTruthActionRecord(
            action=action,
            source=source,
            pred_visible=pred_visible,
            presence_prob=presence_prob,
            frame_id_end=frame_id_end,
            policy_step=policy_step,
        )

    def record_enemy_status(self, can_see: bool, distance: float):
        if self.current is None:
            self.unmatched_status += 1
            return
        if self.current.action in MOVEMENT_ACTIONS:
            self.current.gt_can_see_start = can_see
            self.current.gt_distance_start = distance
        elif self.current.action in SEARCH_ACTIONS:
            self.current.gt_can_see_end = can_see
            self.current.gt_distance_end = distance

    def record_enemy_distance(self, distance: float):
        if self.current is None or self.current.action not in MOVEMENT_ACTIONS:
            self.unmatched_distance += 1
            return
        if self.current.action in MOVEMENT_ACTIONS:
            self.current.gt_distance_end = distance

    def finish_action(self, action: str):
        if self.current is None:
            self.action_mismatch += 1
            return
        if self.current.action != action:
            self.action_mismatch += 1
        self.current.completed = True
        self.records.append(self.current)
        self.current = None

    def abort_action(self, reason: str):
        if self.current is None:
            self.action_mismatch += 1
            return
        self.current.completed = False
        self.current.aborted_reason = reason
        self.records.append(self.current)
        self.current = None

    def reset(self):
        self.current = None
        self.records.clear()
        self.unmatched_status = 0
        self.unmatched_distance = 0
        self.action_mismatch = 0

    def build_summary(self):
        movement_records = [
            record
            for record in self.records
            if (
                record.source == "policy"
                and record.completed
                and record.action in MOVEMENT_ACTIONS
                and record.gt_distance_start is not None
                and record.gt_distance_end is not None
            )
        ]

        search_records = [
            record
            for record in self.records
            if (
                record.source == "policy"
                and record.completed
                and record.action in SEARCH_ACTIONS
                and record.gt_can_see_end is not None
            )
        ]

        perception_records = [
            record
            for record in movement_records
            if (
                record.pred_visible is not None
                and record.gt_can_see_start is not None
            )
        ]

        tp = sum(
            record.pred_visible == 1
            and record.gt_can_see_start is True
            for record in perception_records
        )

        tn = sum(
            record.pred_visible == 0
            and record.gt_can_see_start is False
            for record in perception_records
        )

        fp = sum(
            record.pred_visible == 1
            and record.gt_can_see_start is False
            for record in perception_records
        )

        fn = sum(
            record.pred_visible == 0
            and record.gt_can_see_start is True
            for record in perception_records
        )

        compared = len(perception_records)

        movement_summary = {}
        for action in MOVEMENT_ACTIONS:
            records = [r for r in movement_records if r.action == action]
            deltas = [
                r.gt_distance_start - r.gt_distance_end
                for r in records
            ]
            close_count = sum(delta > 0 for delta in deltas)
            farther_count = sum(delta < 0 for delta in deltas)

            movement_summary[action] = {
                "samples": len(records),
                "closer": close_count,
                "farther": farther_count,
                "mean_delta_distance": sum(deltas) / len(deltas) if deltas else 0.0,
            }

        search_summary = {}
        for action in SEARCH_ACTIONS:
            records = [r for r in search_records if r.action == action]
            visible_after_count = sum(bool(r.gt_can_see_end) for r in records)
            end_distances = [
                r.gt_distance_end for r in records 
                if r.gt_distance_end is not None
            ]

            search_summary[action] = {
                "attempts": len(records),
                "visible_after": visible_after_count,
                "success_rate": visible_after_count / len(records) if records else 0.0,
                "mean_end_distance": sum(end_distances) / len(end_distances) if end_distances else 0.0,
            }

        perception_summary = {
            "compared": compared,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,

            "accuracy": (
                (tp + tn) / compared
                if compared > 0
                else 0.0
            ),

            "precision": (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            ),

            "recall": (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            ),
        }

        return {
            "perception": perception_summary,
            "movement": movement_summary,
            "search": search_summary,

            "diagnostics": {
                "unmatched_status": self.unmatched_status,
                "unmatched_distance": self.unmatched_distance,
                "action_mismatch": self.action_mismatch,
            },
        }

def format_gt_summary(summary: dict) -> str:
    perception = summary["perception"]
    movement = summary["movement"]
    search = summary["search"]
    diagnostics = summary["diagnostics"]

    return (
        "========== Ground Truth Audit ==========\n"

        "Perception:\n"
        f"  Compared: {perception['compared']}\n"
        f"  TP: {perception['tp']} | "
        f"TN: {perception['tn']} | "
        f"FP: {perception['fp']} | "
        f"FN: {perception['fn']}\n"
        f"  Accuracy: {perception['accuracy']:.2%} | "
        f"Precision: {perception['precision']:.2%} | "
        f"Recall: {perception['recall']:.2%}\n"

        "\nMovement:\n"
        f"  Advance:\n"
        f"    Samples: {movement['Advance']['samples']} | "
        f"Closer: {movement['Advance']['closer']} | "
        f"Farther: {movement['Advance']['farther']}\n"
        f"    Mean ΔDistance: "
        f"{movement['Advance']['mean_delta_distance']:+.2f}\n"

        f"  StrafeLeft:\n"
        f"    Samples: {movement['StrafeLeft']['samples']} | "
        f"Closer: {movement['StrafeLeft']['closer']} | "
        f"Farther: {movement['StrafeLeft']['farther']}\n"
        f"    Mean ΔDistance: "
        f"{movement['StrafeLeft']['mean_delta_distance']:+.2f}\n"

        f"  StrafeRight:\n"
        f"    Samples: {movement['StrafeRight']['samples']} | "
        f"Closer: {movement['StrafeRight']['closer']} | "
        f"Farther: {movement['StrafeRight']['farther']}\n"
        f"    Mean ΔDistance: "
        f"{movement['StrafeRight']['mean_delta_distance']:+.2f}\n"

        "\nSearch:\n"
        f"  SearchTurnLeft:\n"
        f"    Visible After: "
        f"{search['SearchTurnLeft']['visible_after']}/"
        f"{search['SearchTurnLeft']['attempts']} | "
        f"Success: "
        f"{search['SearchTurnLeft']['success_rate']:.2%} | "
        f"Mean Distance: "
        f"{search['SearchTurnLeft']['mean_end_distance']:.2f}\n"

        f"  SearchTurnRight:\n"
        f"    Visible After: "
        f"{search['SearchTurnRight']['visible_after']}/"
        f"{search['SearchTurnRight']['attempts']} | "
        f"Success: "
        f"{search['SearchTurnRight']['success_rate']:.2%} | "
        f"Mean Distance: "
        f"{search['SearchTurnRight']['mean_end_distance']:.2f}\n"

        f"  PatrolStepLeft:\n"
        f"    Visible After: "
        f"{search['PatrolStepLeft']['visible_after']}/"
        f"{search['PatrolStepLeft']['attempts']} | "
        f"Success: "
        f"{search['PatrolStepLeft']['success_rate']:.2%} | "
        f"Mean Distance: "
        f"{search['PatrolStepLeft']['mean_end_distance']:.2f}\n"

        f"  PatrolStepRight:\n"
        f"    Visible After: "
        f"{search['PatrolStepRight']['visible_after']}/"
        f"{search['PatrolStepRight']['attempts']} | "
        f"Success: "
        f"{search['PatrolStepRight']['success_rate']:.2%} | "
        f"Mean Distance: "
        f"{search['PatrolStepRight']['mean_end_distance']:.2f}\n"

        "\nDiagnostics:\n"
        f"  Unmatched Status: "
        f"{diagnostics['unmatched_status']}\n"
        f"  Unmatched Distance: "
        f"{diagnostics['unmatched_distance']}\n"
        f"  Action Mismatch: "
        f"{diagnostics['action_mismatch']}\n"

        "========================================"
    )