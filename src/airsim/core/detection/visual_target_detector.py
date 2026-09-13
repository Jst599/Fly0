"""Visual target detection module using multimodal LLM (VLM-only, no OpenCV)"""
import airsim
import numpy as np
import math
import base64
import re
import time
import os
import io
import threading
from queue import Queue
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
from ..ui.ui_utils import Colors

# AirSim Scene 未压缩帧按 BGR 到达（旧实现按 BGR 处理，却把同一份字节当 RGB 编码进 PNG
# 发给模型——PNG 按 R,G,B 解码，模型实际看到的是 R/B 交换后的颜色）。
# 现在定位完全交给 VLM，它是唯一颜色判据，因此这里统一先转成真 RGB 再编码 / 画标注。
# 仅当未来 AirSim 直接返回真 RGB 时才把下面置为 False。
AIRSIM_BGR = True


class VisualTargetDetector:
    """Visual target detector using multimodal LLM (VLM only)"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.is_ollama = "11434" in base_url
        self.openai = None
        # Ollama 走原生 /api/chat，不需要 openai 包；只有 OpenAI 兼容
        # 服务（OpenAI、vLLM、DashScope）才初始化 OpenAI 客户端。
        if not self.is_ollama:
            try:
                from openai import OpenAI
                self.openai = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                print("Warning: openai module not installed for OpenAI-compatible vision endpoint")
            except Exception as e:
                print(f"Warning: OpenAI-compatible vision client init failed: {e}")
        print(f"Visual detector initialized with model: {model} ({'Ollama' if self.is_ollama else 'OpenAI-compatible'})")
        
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 360.0
        self.cy = 240.0
        self._camera_position = None
        self._camera_orientation = None
        self.image_rpc_timeout = 10.0
        self.vlm_timeout = 45.0
        self.vlm_max_attempts = 2
        self.visual_search_timeout = 90.0
        self._last_client = None

    def _start_hover_keepalive(self, client):
        """Keep AirSim's API watchdog satisfied while VLM inference runs."""
        stop = threading.Event()

        def worker():
            while not stop.wait(1.0):
                try:
                    client.hoverAsync(vehicle_name="Drone1")
                except Exception:
                    # The inference result is still useful even if a heartbeat
                    # is lost; the caller will handle the normal RPC error.
                    pass

        thread = threading.Thread(target=worker, daemon=True, name="airsim-vlm-heartbeat")
        thread.start()
        return stop, thread

    def _sim_get_images_with_timeout(self, client, requests):
        """Call AirSim image RPC without allowing the control thread to hang."""
        result_queue = Queue(maxsize=1)

        def worker():
            try:
                result_queue.put((True, client.simGetImages(requests, vehicle_name="Drone1")))
            except Exception as exc:
                result_queue.put((False, exc))

        # The worker is daemonized because a broken msgpack RPC may never
        # return. The main navigation thread remains interruptible.
        threading.Thread(target=worker, daemon=True, name="airsim-image-rpc").start()
        try:
            ok, value = result_queue.get(timeout=self.image_rpc_timeout)
        except Exception:
            raise TimeoutError(
                f"AirSim simGetImages timed out after {self.image_rpc_timeout:.1f}s; "
                "check UE4/AirSim RPC and camera 0"
            )
        if not ok:
            raise RuntimeError(f"AirSim simGetImages failed: {value}")
        return value
    
    def capture_rgbd(self, client: airsim.MultirotorClient) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Capture RGB and depth images"""
        print(f"{Colors.CYAN}[Vision] Requesting RGB-D frame from AirSim camera 0...{Colors.RESET}", flush=True)
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ]
        try:
            responses = self._sim_get_images_with_timeout(client, requests)
        except Exception as e:
            print(f"{Colors.RED}[Vision] {e}{Colors.RESET}", flush=True)
            raise RuntimeError(f"Failed to capture images (is AirSim running?): {e}")
        print(f"{Colors.CYAN}[Vision] RGB-D frame received.{Colors.RESET}", flush=True)
        
        rgb_response = responses[0]
        img_rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb.reshape(rgb_response.height, rgb_response.width, 3)

        # ImageResponse carries the camera pose in AirSim world NED at capture
        # time.  Do not substitute vehicle pose or any UE4 Relative Location.
        self._camera_position = rgb_response.camera_position
        self._camera_orientation = rgb_response.camera_orientation
        try:
            camera_info = client.simGetCameraInfo("0", vehicle_name="Drone1")
            horizontal_fov = math.radians(float(camera_info.fov))
            self.fx = (rgb_response.width / 2.0) / math.tan(horizontal_fov / 2.0)
            self.fy = self.fx
            self.cx = (rgb_response.width - 1) / 2.0
            self.cy = (rgb_response.height - 1) / 2.0
        except Exception as exc:
            print(f"{Colors.YELLOW}Could not read camera FOV; using configured intrinsics: {exc}{Colors.RESET}")
        
        depth_response = responses[1]
        depth_matrix = np.array(depth_response.image_data_float)
        depth_matrix = depth_matrix.reshape(depth_response.height, depth_response.width)
        
        return img_rgb, depth_matrix, rgb_response.height, rgb_response.width
    
    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        """把 AirSim 原始帧转成真 RGB（AIRSIM_BGR=True 时做 BGR->RGB）。返回 C 连续拷贝，PIL 需要。"""
        if AIRSIM_BGR:
            return frame[:, :, ::-1].copy()
        return frame

    def encode_image(self, img_rgb: np.ndarray) -> str:
        """把真 RGB (H,W,3) uint8 数组编码为 base64 PNG 字符串。"""
        buf = io.BytesIO()
        Image.fromarray(img_rgb, mode="RGB").save(buf, format="PNG", compress_level=0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def detect_target_in_rgb(self, img_raw: np.ndarray, user_input: str) -> Tuple[bool, Tuple[int, int]]:
        """纯 VLM 定位：把 AirSim 原始帧转成真 RGB -> PNG base64 发给多模态模型 ->
        解析模型返回的像素 (X,Y)/bbox 或 qwen 原生 bbox_2d(0-1000)，得到目标中心 (列, 行)。"""
        if self.openai is None and not self.is_ollama:
            print(f"{Colors.RED} OpenAI client is None{Colors.RESET}")
            return False, (0, 0)

        try:
            img_rgb = self._to_rgb(img_raw)
            base64_image = self.encode_image(img_rgb)

            prompt = f"""
User command: "{user_input}"

Find the target object in the image based on the complete command.
Return the most relevant target bounding box as JSON:
{{"bbox_2d": [x1, y1, x2, y2]}}
Coordinates use the normalized 0-1000 image scale.
Return {{"bbox_2d": []}} only when the target is not visible.
Return only the JSON object.
"""

            keepalive_stop, keepalive_thread = self._start_hover_keepalive(self._last_client)
            try:
                if self.is_ollama:
                    result = self._detect_with_ollama(prompt, base64_image)
                else:
                    result = self._detect_with_openai(prompt, base64_image)
            finally:
                keepalive_stop.set()
                keepalive_thread.join(timeout=0.2)

            print(f"{Colors.CYAN}LLM原始响应: {result}{Colors.RESET}")

            height, width = img_rgb.shape[0], img_rgb.shape[1]
            found, x, y, box = self._parse_vlm_result(result, width, height)

            if found:
                path = self._save_marked_png(img_rgb, x, y, box)
                print(f"{Colors.GREEN}Target found at center: ({x},{y}), saved to {path}{Colors.RESET}")
                return True, (x, y)

            print(f"{Colors.YELLOW} No target found in result{Colors.RESET}")
            return False, (0, 0)
        except Exception as e:
            print(f"{Colors.RED}Exception in detect_target_in_rgb: {str(e)}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False, (0, 0)

    def _parse_vlm_result(self, result: str, width: int, height: int) -> Tuple[bool, int, int, Optional[Tuple[int, int, int, int]]]:
        """解析 VLM 定位结果，兼容三种格式：
        (a) qwen 原生 grounding——JSON bbox_2d 或裸 [x1,y1,x2,y2]，0-1000 归一化；
        (b) 像素框 "(x1,y1,x2,y2)"；(c) 像素中心 "(X,Y)"。
        兜底规则：数值落在图像外、却都在 0-1000 内 → 按 qwen 原生 0-1000 归一化重算
        （qwen3-vl 常无视"720*480 像素"直接按其原生刻度报中心，如 Y=500）；
        换算后中心仍不在图像内 → 视为未找到，绝不返回画布外的坐标。
        返回 (是否找到, 中心列x, 中心行y, 像素框或None)。中心 (0,0) / [] / [0,0,0,0] / 文本 not found 视为未找到。"""
        # 1) 未找到守卫
        if not result:
            return False, 0, 0, None
        stripped = result.strip()
        if stripped in ("[]", "{}", "(0,0)", "(0, 0)"):
            return False, 0, 0, None
        if "not found" in stripped.lower():
            return False, 0, 0, None

        # 2) 格式(a)：qwen 原生 bbox_2d（0-1000 归一化）——优先，避免被像素解析误吞
        m = re.search(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', result)
        if m is None:
            m = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', result)
        if m:
            vals = [float(g) for g in m.groups()]
            if all(v == 0 for v in vals):
                return False, 0, 0, None
            return self._center_from_normalized_box(vals, width, height)

        # 3) 格式(b)：像素框 "(x1,y1,x2,y2)"；无法作为像素框但数值都在 0-1000 → 归一化框
        m = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', result)
        if m:
            x1, y1, x2, y2 = map(int, m.groups())
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                return False, 0, 0, None
            pixel_ok = (0 <= x1 <= x2 < width) and (0 <= y1 <= y2 < height)
            if not pixel_ok and 0 <= min(x1, y1, x2, y2) and max(x1, y1, x2, y2) <= 1000:
                return self._center_from_normalized_box((x1, y1, x2, y2), width, height)
            if not pixel_ok:
                return False, 0, 0, None
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return True, cx, cy, (x1, y1, x2, y2)

        # 4) 格式(c)：像素中心 "(X,Y)"；Y=500 这类越界但 0-1000 内 → qwen 原生中心，重算
        m = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', result)
        if m:
            x, y = map(int, m.groups())
            if x == 0 and y == 0:
                return False, 0, 0, None
            if not (0 <= x < width and 0 <= y < height):
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    cx = int(round(x / 1000.0 * width))
                    cy = int(round(y / 1000.0 * height))
                    if not (0 <= cx < width and 0 <= cy < height) or (cx == 0 and cy == 0):
                        return False, 0, 0, None
                    return True, cx, cy, None
                return False, 0, 0, None
            return True, x, y, None

        return False, 0, 0, None

    @staticmethod
    def _center_from_normalized_box(vals, width: int, height: int):
        """把 qwen 原生 0-1000 归一化框换算成像素框与像素中心。中心落在图外 → 判未找到。"""
        x1, y1, x2, y2 = [float(v) for v in vals]
        cx = int(round((x1 + x2) / 2.0 / 1000.0 * width))
        cy = int(round((y1 + y2) / 2.0 / 1000.0 * height))
        if cx == 0 and cy == 0:
            return False, 0, 0, None
        if not (0 <= cx < width and 0 <= cy < height):
            return False, 0, 0, None
        box_px = (int(round(x1 / 1000.0 * width)), int(round(y1 / 1000.0 * height)),
                  int(round(x2 / 1000.0 * width)), int(round(y2 / 1000.0 * height)))
        return True, cx, cy, box_px

    def _save_marked_png(self, rgb_img: np.ndarray, x: int, y: int,
                         box: Optional[Tuple[int, int, int, int]] = None) -> str:
        """在真 RGB 帧上画调试标注（红圈 + 绿心点 + 可选蓝框）并保存 PNG，返回路径。"""
        os.makedirs("images", exist_ok=True)
        path = os.path.join("images", f"detect_{int(time.time())}_marked.png")
        pil_img = Image.fromarray(rgb_img, mode="RGB")
        draw = ImageDraw.Draw(pil_img)
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 0, 0), width=2)  # 红圈
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))                   # 绿心点
        if box is not None:
            draw.rectangle(list(box), outline=(0, 0, 255), width=2)                    # 蓝框
        pil_img.save(path, format="PNG")
        return path

    def _detect_with_ollama(self, prompt: str, base64_image: str) -> str:
        """Detect target using Ollama API"""
        import requests
        import json
        
        # Keep the model's native Qwen3-VL response format.  The Ollama
        # ``format=json`` constraint caused this GGUF build to stop with an
        # empty ``content`` field even when it had a valid visual answer.
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            # Disable thinking where supported, but let the model emit its
            # normal grounding JSON in message.content.
            "think": False,
            "options": {"temperature": 0, "num_predict": 512}
        }
        
        print(f"{Colors.CYAN}调用Ollama API，模型: {self.model}{Colors.RESET}")
        print(f"{Colors.CYAN}Base64图像长度: {len(base64_image)}{Colors.RESET}")
        
        try:
            response = None
            for attempt in range(self.vlm_max_attempts):
                print(f"{Colors.CYAN}[Vision] Ollama request attempt {attempt + 1}/{self.vlm_max_attempts}...{Colors.RESET}", flush=True)
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=data,
                    timeout=(5, self.vlm_timeout)
                )
                if response.status_code in (502, 503, 504) and attempt + 1 < self.vlm_max_attempts:
                    time.sleep(2)
                    continue
                if response.status_code == 200:
                    payload = response.json()
                    content = payload.get("message", {}).get("content", "") or ""
                    done_reason = payload.get("done_reason", "unknown")
                    thinking = payload.get("message", {}).get("thinking", "") or ""
                    # Some Ollama/Qwen3-VL builds put the final grounding JSON
                    # in message.thinking even with think=false.  Accept only
                    # the bbox JSON fragment, never arbitrary reasoning text.
                    if not content and thinking:
                        bbox = re.search(
                            r'\{\s*"bbox_2d"\s*:\s*\[[^\]]*\]\s*\}',
                            thinking,
                        )
                        if bbox:
                            content = bbox.group(0)
                            payload.setdefault("message", {})["content"] = content
                    if not content and done_reason == "length" and attempt + 1 < self.vlm_max_attempts:
                        print(
                            f"{Colors.YELLOW}[Vision] model output was truncated after "
                            f"{len(thinking)} thinking characters; retrying with num_predict=1024.{Colors.RESET}"
                        )
                        data["options"]["num_predict"] = 1024
                        continue
                break
            
            print(f"{Colors.CYAN}Ollama API响应状态码: {response.status_code}{Colors.RESET}")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {}) or {}
                content = message.get("content", "") or ""
                thinking = message.get("thinking", "") or ""
                if not content and thinking:
                    bbox = re.search(
                        r'\{\s*"bbox_2d"\s*:\s*\[[^\]]*\]\s*\}',
                        thinking,
                    )
                    if bbox:
                        content = bbox.group(0)
                print(
                    f"{Colors.CYAN}[Vision] done_reason={result.get('done_reason', 'unknown')}, "
                    f"thinking_chars={len(thinking)}{Colors.RESET}"
                )
                print(f"{Colors.CYAN}Ollama API响应内容: {content}{Colors.RESET}")
                return content
            else:
                print(f"{Colors.RED}[ERROR] Ollama API error: {response.status_code}{Colors.RESET}")
                print(f"{Colors.RED}[ERROR] Response: {response.text}{Colors.RESET}")
                return ""
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Ollama API request failed: {str(e)}{Colors.RESET}")
            return ""
    
    def _detect_with_openai(self, prompt: str, base64_image: str) -> str:
        """Detect target using OpenAI-compatible API"""
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=64,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def calculate_target_position(self, img_coords: Tuple[int, int], depth_matrix: np.ndarray,
                                  client: airsim.MultirotorClient) -> Tuple[float, float, float]:
        """Calculate target world coordinates"""
        u, v = img_coords
        height, width = depth_matrix.shape

        if v >= height or u >= width or v < 0 or u < 0:
            return (0, 0, 0)

        MAX_DEPTH = 50.0  # 只接受 50 米以内的深度

        radius_pixels = 10  # 扩大到 10 像素半径采样
        valid_depths = []

        min_x = max(0, int(u - radius_pixels))
        max_x = min(width - 1, int(u + radius_pixels))
        min_y = max(0, int(v - radius_pixels))
        max_y = min(height - 1, int(v + radius_pixels))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.sqrt((x - u) ** 2 + (y - v) ** 2)
                if distance <= radius_pixels:
                    depth = depth_matrix[y, x]
                    if 0.1 < depth < MAX_DEPTH and not math.isinf(depth) and not math.isnan(depth):
                        valid_depths.append(depth)

        if not valid_depths:
            # 调试：打印这个像素区域的实际深度值
            sample_depths = []
            for sy in range(max(0, v-3), min(height, v+4)):
                for sx in range(max(0, u-3), min(width, u+4)):
                    sample_depths.append(f"({sx},{sy}):{depth_matrix[sy, sx]:.2f}")
            print(f"{Colors.RED}✗ No valid depth at ({u},{v}). Nearby depths: {', '.join(sample_depths[:9])}{Colors.RESET}")
            return (0, 0, 0)

        # 用最小深度——球面最近点才是目标真实距离（背景/边缘深度会偏大）
        depth = min(valid_depths)
        print(f"{Colors.CYAN}Depth at target: {depth:.2f}m (sampled {len(valid_depths)} points){Colors.RESET}")
        
        X_cam = depth
        Y_cam = (u - self.cx) * depth / self.fx
        Z_cam = (v - self.cy) * depth / self.fy
        
        camera_position = self._camera_position
        camera_orientation = self._camera_orientation
        if camera_position is None or camera_orientation is None:
            camera_info = client.simGetCameraInfo("0", vehicle_name="Drone1")
            camera_position = camera_info.pose.position
            camera_orientation = camera_info.pose.orientation
        
        q = np.array([camera_orientation.x_val, camera_orientation.y_val,
                      camera_orientation.z_val, camera_orientation.w_val])
        rotation_matrix = self._quaternion_to_rotation_matrix(q)
        
        camera_point = np.array([X_cam, Y_cam, Z_cam])
        world_point = np.dot(rotation_matrix, camera_point)
        
        X_world = camera_position.x_val + world_point[0]
        Y_world = camera_position.y_val + world_point[1]
        Z_world = camera_position.z_val + world_point[2]
        
        return X_world, Y_world, Z_world
    
    def detect_visual_target(self, client: airsim.MultirotorClient, user_input: str, 
                            search_360: bool = True) -> Tuple[bool, Tuple[float, float, float]]:
        """Detect visual target"""
        self._last_client = client
        search_deadline = time.time() + self.visual_search_timeout
        print(f"{Colors.CYAN}[Vision] Detecting target: {user_input}{Colors.RESET}", flush=True)
        img_rgb, depth_matrix, height, width = self.capture_rgbd(client)
        print(f"{Colors.CYAN}[Vision] Sending first frame to target detector...{Colors.RESET}", flush=True)
        target_found, img_coords = self.detect_target_in_rgb(img_rgb, user_input)
        
        if not target_found and search_360:
            print("Target not found, starting 360-degree search...")
            search_steps = 4
            rotation_angle = 360 / search_steps
            
            for i in range(search_steps):
                if time.time() >= search_deadline:
                    print(f"{Colors.YELLOW}[Vision] visual search timeout; stopping rotation search{Colors.RESET}")
                    break
                print(f"{Colors.CYAN}[Vision] 360 search step {i + 1}/{search_steps}...{Colors.RESET}", flush=True)
                self.rotate_drone(client, rotation_angle)
                img_rgb, depth_matrix, height, width = self.capture_rgbd(client)
                target_found, img_coords = self.detect_target_in_rgb(img_rgb, user_input)
                
                if target_found:
                    print(f"Target found after {i+1} rotations")
                    break
        
        if not target_found:
            print("Target not found")
            return False, (0, 0, 0)
        
        print(f"Target image coordinates: {img_coords}")
        
        target_pos = self.calculate_target_position(img_coords, depth_matrix, client)
        
        return True, target_pos
    
    def rotate_drone(self, client: airsim.MultirotorClient, degrees: float, velocity: float = 30.0):
        """Rotate drone in place"""
        current_pose = client.simGetVehiclePose()
        current_yaw = math.atan2(
            2.0 * (current_pose.orientation.w_val * current_pose.orientation.z_val +
                   current_pose.orientation.x_val * current_pose.orientation.y_val),
            1.0 - 2.0 * (current_pose.orientation.y_val ** 2 + current_pose.orientation.z_val ** 2)
        )
        # AirSim rotateToYawAsync expects degrees, while atan2 returns radians.
        target_yaw_degrees = math.degrees(current_yaw) + degrees
        timeout = max(2.0, abs(degrees) / max(velocity, 1e-3) + 2.0)
        client.rotateToYawAsync(
            target_yaw_degrees,
            timeout_sec=timeout,
            margin=5.0,
            vehicle_name="Drone1"
        ).join()
        time.sleep(1)
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
