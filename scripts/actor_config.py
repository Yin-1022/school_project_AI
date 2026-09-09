from dataclasses import dataclass

@dataclass
class ActorConfig:
    actor_id: str = "actor_0"
    frame_host: str = "127.0.0.1"
    frame_port: int = 9999
    ue_host: str = "127.0.0.1"
    action_port: int = 12345
    event_port: int = 12346