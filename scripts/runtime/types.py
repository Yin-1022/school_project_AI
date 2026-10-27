from dataclasses import dataclass
from enum import Enum


class ActionSource(Enum):
    POLICY = "policy"
    RUNTIME_RECOVERY = "runtime_recovery"
    RUNTIME_HIT_REACTION = "runtime_hit_reaction"
    RUNTIME_FALLBACK = "runtime_fallback"


@dataclass(frozen=True)
class RuntimeAction:
    name: str
    value: float
    source: ActionSource


@dataclass
class ExecutionState:
    busy: bool = False
    waiting_start: bool = False
    current_action: RuntimeAction | None = None
    sent_at: float | None = None
    started_at: float | None = None


class ExecutionEndReason(str, Enum):
    COMPLETED = "completed"
    START_TIMEOUT = "actstart_timeout"
    END_TIMEOUT = "actend_timeout"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionResult:
    action: RuntimeAction
    reason: ExecutionEndReason