
import pytest

from ground_truth_audit import GroundTruthAudit
from runtime.types import RuntimeAction, ActionSource
from runtime.adapters.impala_adapter import to_runtime_action


@pytest.mark.parametrize(
    "action_name",
    [
        "Advance",
        "StrafeLeft",
        "SearchTurnLeft",
        "SearchTurnRight",
        "PatrolStepLeft",
        "PatrolStepRight",
    ],
)
def test_policy_action_runtime_mapping_matches(action_name):
    audit = GroundTruthAudit()
    expected = to_runtime_action(action_name)

    audit.begin_action(
        action=action_name,
        source="policy",
        expected_runtime_action=expected,
    )

    audit.finish_action(expected)

    assert audit.action_mismatch == 0
    assert len(audit.records) == 1
    assert audit.records[0].action == action_name
    assert audit.records[0].completed is True


def test_runtime_action_wrong_angle_is_mismatch():
    audit = GroundTruthAudit()

    audit.begin_action(
        action="SearchTurnLeft",
        source="policy",
        expected_runtime_action=to_runtime_action("SearchTurnLeft"),
    )

    audit.finish_action(
        RuntimeAction("SearchTurn", 50.0, ActionSource.POLICY)
    )

    assert audit.action_mismatch == 1
    assert audit.records[0].completed is False
    assert "finish mismatch" in audit.records[0].aborted_reason


def test_runtime_action_wrong_source_is_mismatch():
    audit = GroundTruthAudit()

    audit.begin_action(
        action="SearchTurn",
        source="recovery",
        expected_runtime_action=RuntimeAction(
            "SearchTurn", 90.0, ActionSource.RUNTIME_RECOVERY
        ),
    )

    audit.finish_action(
        RuntimeAction(
            "SearchTurn", 90.0, ActionSource.RUNTIME_HIT_REACTION
        )
    )

    assert audit.action_mismatch == 1
    assert audit.records[0].completed is False


def test_runtime_bound_audit_cannot_self_validate_with_string():
    audit = GroundTruthAudit()

    audit.begin_action(
        action="SearchTurnLeft",
        source="policy",
        expected_runtime_action=to_runtime_action("SearchTurnLeft"),
    )

    audit.finish_action("SearchTurnLeft")

    assert audit.action_mismatch == 1
    assert audit.records[0].completed is False


def test_finish_without_begin_counts_mismatch():
    audit = GroundTruthAudit()

    audit.finish_action(
        RuntimeAction("Advance", 0.0, ActionSource.POLICY)
    )

    assert audit.action_mismatch == 1
    assert audit.current is None
