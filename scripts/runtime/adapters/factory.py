
from dataclasses import dataclass
from collections.abc import Callable, Mapping

from runtime.types import RuntimeAction


@dataclass(frozen=True)
class ModelAdapter:
    to_runtime_action: Callable[[str], RuntimeAction]
    to_osc_payload: Callable[[RuntimeAction], object]


def create_adapter(
    model: str,
    *,
    action_values: Mapping[str, float] | None = None,
) -> ModelAdapter:

    if model == "IMPALA":
        from runtime.adapters.impala_adapter import (
            to_runtime_action,
            to_osc_payload,
        )

        return ModelAdapter(
            to_runtime_action=to_runtime_action,
            to_osc_payload=to_osc_payload,
        )

    elif model == "A3C":
        from runtime.adapters.a3c_adapter import A3CAdapter

        if action_values is None:
            raise ValueError("A3C requires action_values")

        adapter = A3CAdapter(action_values)

        return ModelAdapter(
            to_runtime_action=adapter.to_runtime_action,
            to_osc_payload=adapter.to_osc_payload,
        )

    else:
        raise ValueError(f"Unsupported runtime model: {model}")
