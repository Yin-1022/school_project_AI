from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient
import threading
import socket
import numpy as np
import cv2

_OSC_CLIENT = None

def receive_from_ue(UE_EVENT_LOCK, UE_EVENT_STATE):
    def on_attack_start(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["attack_active"] = True
            UE_EVENT_STATE["attack_start_pulse"] = True
        print(f"[← UE] 開始攻擊！args: {args}\n")

    def on_attack_end(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["attack_active"] = False
            UE_EVENT_STATE["attack_end_pulse"] = True
        print(f"[← UE] 結束攻擊！args: {args}\n")

    def on_fallback(address, *args):
        print(f"[← UE] 未知訊息 {address}，args: {args}\n")

    def on_boss_hit(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["boss_hit_pulse"] = True
        print(f"[← UE] Boss 被擊中！args: {args}\n")

    def on_health_changed(address, *args):
        health = args[0]
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["player_hit_pulse"] = True
        print(f"[← UE] 玩家血量：{health}")

    def on_episode_done(address, *args):
        with UE_EVENT_LOCK:
            UE_EVENT_STATE["episode_done_flag"] = True
        print(f"[← UE] 回合結束！args: {args}\n")

    dp = dispatcher.Dispatcher()
    dp.map("/attatart", on_attack_start)
    dp.map("/attend",   on_attack_end)
    dp.map("/enemy_take_damage", on_boss_hit)
    dp.map("/player_health", on_health_changed)
    dp.map("/game_over", on_episode_done)
    dp.set_default_handler(on_fallback)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 12346), dp)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[接收] 監聽 port {12346}...")

def get_osc_client():
    global _OSC_CLIENT
    if _OSC_CLIENT is None:
        _OSC_CLIENT = SimpleUDPClient("127.0.0.1", 12345)
    return _OSC_CLIENT

def send_action(msg):
    client = get_osc_client()

    action_name = msg["action"]
    angle = 0.0

    if action_name == "SearchTurnRight":
        action_name = "SearchTurn"
        angle = 50.0
    elif action_name == "SearchTurnLeft":
        action_name = "SearchTurn"
        angle = -50.0
    elif action_name == "PatrolStepRight":
        action_name = "PatrolStep"
        angle = 50.0
    elif action_name == "PatrolStepLeft":
        action_name = "PatrolStep"
        angle = -50.0

    args = [
        action_name,                            # string
        float(angle),                           # float
        int(msg["ts_frame"]),                   # int
        int(msg["fire_frame"]),                 # int
        int(msg["hold_until"]),                 # int
        float(msg["meta"]["conf"]),             # float
        str(msg["meta"]["phase"]),              # string
        str(msg["meta"]["search_hint"] or ""),  # string
        int(msg["seq"]),                        # int
    ]

    client.send_message("/boss/action", args)

def tcp_frame_stream(host='127.0.0.1', port=9999, img_w=192, img_h=192, img_c=3, debug_show=False):
    frame_size = img_w * img_h * img_c

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("=== Python 推論伺服器已就緒 ===")

    while True:
        print(f"[等待中] 正在監聽 Port {port}...")
        conn = None
        try:
            conn, addr = server_socket.accept()
            print(f"[已連線] 與 UE 建立連線: {addr}")

            data_buffer = b""
            while True:
                while len(data_buffer) < frame_size:
                    packet = conn.recv(frame_size - len(data_buffer))
                    if not packet:
                        print("[通知] UE 連線中斷")
                        raise ConnectionResetError
                    data_buffer += packet

                frame_data = data_buffer[:frame_size]
                data_buffer = data_buffer[frame_size:]

                frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((img_h, img_w, img_c))

                # 模型若沿用 OpenCV 訓練資料，建議轉回 BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                conn.sendall(b"OK")
                yield frame_bgr

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print("[系統] 目前連線已中斷，回到監聽狀態")
            yield None
        except KeyboardInterrupt:
            print("[系統] 收到中斷事件，忽略這次中斷並回到監聽狀態")
        finally:
            if conn is not None:
                conn.close()
            if debug_show:
                cv2.destroyAllWindows()