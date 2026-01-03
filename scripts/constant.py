CLIP_FRAMES     = 8          
CLIP_STRIDE     = 4          
TARGET_FPS      = 12
FRAME_SIZE      = (192, 192)
ms_per_frame    = int(1000 / TARGET_FPS)
delta_t_ms      = CLIP_STRIDE * ms_per_frame

TAU_CMD = 0.60
MIN_HOLD_FRAMES = round(0.3 * TARGET_FPS)
RT_FRAMES = round(0.2 * TARGET_FPS)
CD_EVADE = round(0.8 * TARGET_FPS)
CD_TURN = round(0.3 * TARGET_FPS)
CD_PATROL = round(0.5 * TARGET_FPS)
