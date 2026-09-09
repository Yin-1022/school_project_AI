import argparse
from dataclasses import dataclass

@dataclass
class ActorConfig:
    actor_id: str = "actor_0"
    frame_host: str = "127.0.0.1"
    frame_port: int = 9999
    ue_host: str = "127.0.0.1"
    action_port: int = 12345
    event_port: int = 12346

def parse_actor_config() -> ActorConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--actor-id", default="actor_0")
    parser.add_argument("--frame-host", default="127.0.0.1")
    parser.add_argument("--frame-port", type=int, default=9999)
    parser.add_argument("--ue-host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=12345)
    parser.add_argument("--event-port", type=int, default=12346)

    args = parser.parse_args()

    return ActorConfig(
        actor_id=args.actor_id,
        frame_host=args.frame_host,
        frame_port=args.frame_port,
        ue_host=args.ue_host,
        action_port=args.action_port,
        event_port=args.event_port,
    )