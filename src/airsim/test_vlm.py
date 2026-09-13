"""VLM 纯视觉冒烟验证 - 拍照 → Ollama(qwen2.5vl:7b) 定位 → PIL 标注保存

用法：先起 UE(AirSim) 和 Ollama，再运行（可用参数覆盖目标指令）::

    python test_vlm.py ["Move to the red ball"]
"""
import airsim
import numpy as np
import os
import sys
import threading
import time
from PIL import Image

from core.detection.visual_target_detector import VisualTargetDetector

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5vl:7b"

USER_INPUT = sys.argv[1] if len(sys.argv) > 1 else "Move to the red ball"

print("1/4 连接 AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ 已连接\n")


def capture_with_timeout(client, requests, timeout=15):
    result = [None]
    exc = [None]

    def _capture():
        try:
            result[0] = client.simGetImages(requests, vehicle_name="Drone1")
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_capture, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"   ⚠ 拍照超时 ({timeout}s)，AirSim RPC 僵死")
        print("   → 先 UE Stop→Play，等 5 秒，再重新运行此脚本")
        sys.exit(1)
    if exc[0]:
        raise exc[0]
    return result[0]


print("2/4 拍照（相机 0 Scene + DepthPerspective）...")
responses = capture_with_timeout(client, [
    airsim.ImageRequest("0", 0, False, False),
    airsim.ImageRequest("0", 2, True, False),
], timeout=15)
print(f"   simGetImages 返回 {len(responses)} 个响应")
for i, r in enumerate(responses):
    print(f"   [{i}] image_type={r.image_type}, width={r.width}, height={r.height}")

img_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
    responses[0].height, responses[0].width, 3)
h, w = img_raw.shape[:2]
depth = None
if len(responses) > 1 and responses[1].image_data_float:
    depth = np.array(responses[1].image_data_float).reshape(
        responses[1].height, responses[1].width)
    print(f"   尺寸: {w}x{h}, 深度: {depth.min():.1f}~{depth.max():.1f}m\n")
else:
    print(f"   尺寸: {w}x{h}\n")

os.makedirs("images", exist_ok=True)
ts = int(time.time())

print(f"3/4 纯 VLM 定位（目标指令: {USER_INPUT}）...")
# 复用运行时检测器（同一套 编码/提示词/解析），保证冒烟与真机一致
det = VisualTargetDetector(api_key="", base_url=OLLAMA_URL, model=MODEL)
found, (x, y) = det.detect_target_in_rgb(img_raw, USER_INPUT)

# 保留一张真 RGB 原图，便于和标注图对照颜色
Image.fromarray(det._to_rgb(img_raw), mode="RGB").save(f"images/test_{ts}_raw.png")
print(f"   真RGB原图已保存: images/test_{ts}_raw.png")

if found:
    print(f"   模型定位中心: ({x}, {y})")
    # 用坐标周围半径 15 像素内的最小有效深度（球最近点才是球面真实距离）
    d = -1
    if depth is not None:
        valid = []
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dv = depth[ny, nx]
                    if 0.1 < dv < 100:  # 过滤无效深度
                        valid.append(dv)
        if valid:
            d = min(valid)
    if 0.1 < d < 50:
        print(f"   ✓ 目标距相机约 {d:.1f} 米")
    elif 0 <= x < w and 0 <= y < h:
        print("   ✓ 坐标在画面内（无有效深度，跳过距离）")
    else:
        print(f"   ⚠ 坐标({x},{y})超出图像范围({w}x{h})")
    print("   标注图已由 detector 保存：images/detect_*_marked.png")
else:
    print("   ⚠ 模型报告：画面中无目标（返回 (0,0) 或 []）")

print("\n完成！查看 images/ 文件夹。")
