from time import monotonic

class AttackGuard:
    def __init__(
            self,
            duration: dict[str, float] | None = None,
            margin: float = 0.15,
    ):
        if duration is None:
            duration = {
                "att1": 52/30.0,
                "att2": 69/30.0,
            }
        self.duration = duration
        self.margin = margin
        self.active_deadline: dict[str, float] = {}

    def on_attack_start(self, attack_type: str) -> None:
        if attack_type not in self.duration:
            raise ValueError(f"Unknown attack type: {attack_type}")
        if attack_type in self.active_deadline:
            return
        self.active_deadline[attack_type] = monotonic() + self.duration[attack_type] + self.margin

    def on_attack_end(self, attack_type: str) -> None:
        if attack_type in self.active_deadline:
            del self.active_deadline[attack_type]

    def is_attacking(self) -> bool:
        return bool(self.active_deadline)

    def check_timeouts(self, now=None) -> list[str]:
        if now is None:
            now = monotonic()

        expired_attacks = [attack for attack, deadline in self.active_deadline.items() 
                           if now >= deadline]

        for attack in expired_attacks:
            del self.active_deadline[attack]

        return expired_attacks

    def reset(self) -> None:
        self.active_deadline.clear()