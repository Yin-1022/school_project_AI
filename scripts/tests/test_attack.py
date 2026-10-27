from time import monotonic
from runtime.attack import AttackGuard


def test_default_attack_guard():
    guard = AttackGuard()
    assert not guard.is_attacking()

def test_att1_start_and_end():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    assert guard.is_attacking()
    guard.on_attack_end("att1")
    assert not guard.is_attacking()

def test_both_attack_with_only_one_end():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    guard.on_attack_start("att2")
    assert guard.is_attacking()
    guard.on_attack_end("att1")
    assert guard.is_attacking()

def test_both_attack_with_only_one_timeout():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    guard.on_attack_start("att2")
    deadline = guard.active_deadline["att1"]

    assert guard.check_timeouts(now=deadline - 0.1) == []
    assert guard.check_timeouts(now=deadline) == ["att1"]
    assert guard.is_attacking()
    assert "att1" not in guard.active_deadline
    assert "att2" in guard.active_deadline

def test_unknown_attack_type():
    guard = AttackGuard()
    try:
        guard.on_attack_start("unknown_attack")
    except ValueError as e:
        assert str(e) == "Unknown attack type: unknown_attack"
    else:
        assert False, "Expected ValueError for unknown attack type"

def test_no_exception_on_repeated_end():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    guard.on_attack_end("att1")
    try:
        guard.on_attack_end("att1")  # Should not raise an exception
    except Exception as e:
        assert False, f"Expected no exception, but got: {e}"

def test_no_extending_deadline_on_repeated_start():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    first_deadline = guard.active_deadline["att1"]
    guard.on_attack_start("att1")  # Repeated start should not extend the deadline
    assert guard.active_deadline["att1"] == first_deadline

def test_reset():
    guard = AttackGuard()
    guard.on_attack_start("att1")
    guard.on_attack_start("att2")
    assert guard.is_attacking()
    guard.reset()
    assert not guard.is_attacking()