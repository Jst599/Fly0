import time
from typing import List, Optional

import airsim
import numpy as np


class DroneController:
    """Deterministic AirSim wrapper using Fly0 [x, y, height] coordinates."""

    def __init__(self, drone_name: str = "Drone1", takeoff_reference_ned_z: Optional[float] = None):
        self.drone_name = drone_name
        self.takeoff_reference_ned_z = (
            float(takeoff_reference_ned_z)
            if takeoff_reference_ned_z is not None else None
        )
        self.client = airsim.MultirotorClient()
        self._initialize()

    def _initialize(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        try:
            self.client.rotateToYawAsync(0.0, 5.0, vehicle_name=self.drone_name).join()
        except Exception:
            pass

    def get_relative_altitude(self) -> float:
        absolute_height = self.get_position()[2]
        if self.takeoff_reference_ned_z is None:
            return absolute_height
        return absolute_height + self.takeoff_reference_ned_z

    def relative_altitude_target(self, altitude: float) -> float:
        if self.takeoff_reference_ned_z is None:
            return float(altitude)
        return -self.takeoff_reference_ned_z + float(altitude)

    def takeoff(self):
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()

    def land(self):
        self.client.landAsync(vehicle_name=self.drone_name).join()

    def hover(self):
        self.client.hoverAsync(vehicle_name=self.drone_name).join()

    def fly_to(self, point: List[float], velocity: float = 5.0, timeout: float = 60.0) -> bool:
        """Move to [x, y, height], verify the real pose, and recover only on collision."""
        if len(point) != 3:
            raise ValueError("fly_to expects [x, y, height]")
        target = np.asarray(point, dtype=float)
        collision_baseline = self._collision_timestamp()
        collision_candidate = None
        self.client.moveToPositionAsync(
            float(target[0]), float(target[1]), -float(target[2]),
            max(0.1, float(velocity)), vehicle_name=self.drone_name
        )

        arrival_tolerance = 0.8
        stable_time = 0.6
        stall_time = 8.0
        deadline = time.time() + max(1.0, float(timeout))
        last_position = np.asarray(self.get_position(), dtype=float)
        last_error = float(np.linalg.norm(last_position - target))
        last_progress_time = time.time()
        stable_since = None
        collision_detected = False

        while time.time() < deadline:
            collision_event = self._new_collision_event(collision_baseline)
            if collision_event is None:
                collision_candidate = None
            elif collision_candidate is not None and collision_event[1] == collision_candidate[1]:
                print(
                    f"Confirmed collision: object={collision_event[1] or '<unknown>'}, "
                    f"timestamp={collision_event[0]:.0f}"
                )
                print("检测到碰撞，取消当前任务并紧急上升 30.0m")
                self._cancel_and_escape()
                collision_detected = True
                break
            else:
                collision_candidate = collision_event

            actual = np.asarray(self.get_position(), dtype=float)
            error = float(np.linalg.norm(actual - target))
            speed = float(np.linalg.norm(self.get_velocity()))
            if error <= arrival_tolerance and speed <= 1.0:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_time:
                    break
            else:
                stable_since = None

            moved = float(np.linalg.norm(actual - last_position))
            progress = last_error - error
            if moved >= 0.05 or progress >= 0.03:
                last_position = actual
                last_error = error
                last_progress_time = time.time()
            elif time.time() - last_progress_time >= stall_time and error > 2.0:
                print(
                    f"无人机无碰撞证据且 {stall_time:.0f} 秒无有效进展，"
                    f"取消当前任务（误差 {error:.2f}m），不执行紧急上升"
                )
                self._cancel_task()
                break
            time.sleep(0.1)

        actual = np.asarray(self.get_position(), dtype=float)
        error = float(np.linalg.norm(actual - target))
        if time.time() >= deadline and error > arrival_tolerance:
            print("直接飞行超时，取消当前任务")
            self._cancel_task()
        print(
            f"实际位置: [{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}] | "
            f"目标误差: {error:.3f} m"
        )
        return (not collision_detected) and error <= arrival_tolerance

    def _collision_count(self) -> int:
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
            return int(getattr(info, "collision_count", 0) or 0)
        except Exception:
            return 0

    def _collision_timestamp(self) -> float:
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
            return float(getattr(info, "time_stamp", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _new_collision_event(self, baseline_timestamp: float):
        """Return a signature only for a collision after this move began."""
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        except Exception:
            return None
        if not bool(getattr(info, "has_collided", False)):
            return None
        timestamp = float(getattr(info, "time_stamp", 0.0) or 0.0)
        if timestamp <= float(baseline_timestamp):
            return None
        return (timestamp, str(getattr(info, "object_name", "") or ""))

    def _cancel_task(self):
        try:
            self.client.cancelLastTask()
        except Exception as exc:
            print(f"AirSim cancelLastTask failed: {exc}")

    def _cancel_and_escape(self, climb_meters: float = 30.0, climb_speed: float = 3.0):
        self._cancel_task()
        try:
            duration = abs(float(climb_meters)) / max(abs(float(climb_speed)), 0.1)
            self.client.moveByVelocityAsync(
                0.0, 0.0, -abs(float(climb_speed)), duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.drone_name
            ).join()
            self.client.hoverAsync(vehicle_name=self.drone_name).join()
        except Exception as exc:
            print(f"紧急脱困失败，AirSim 可能已进入安全悬停: {exc}")

    def fly_path(self, points: List[List[float]], velocity: float = 5.0):
        if not points:
            return
        airsim_points = [
            airsim.Vector3r(float(p[0]), float(p[1]), -float(p[2]))
            for p in points
        ]
        self.client.moveOnPathAsync(
            airsim_points, max(0.1, float(velocity)), 120,
            airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),
            20, 1, vehicle_name=self.drone_name
        ).join()

    def set_yaw(self, yaw: float, timeout: float = 5.0):
        self.client.rotateToYawAsync(float(yaw), float(timeout), vehicle_name=self.drone_name).join()

    def get_position(self) -> List[float]:
        pose = self.client.simGetVehiclePose(vehicle_name=self.drone_name)
        return [pose.position.x_val, pose.position.y_val, -pose.position.z_val]

    def get_yaw(self) -> float:
        q = self.client.simGetVehiclePose(vehicle_name=self.drone_name).orientation
        yaw_rad = np.arctan2(
            2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
            1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
        )
        return float(np.degrees(yaw_rad) % 360.0)

    def get_velocity(self) -> np.ndarray:
        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        v = state.kinematics_estimated.linear_velocity
        return np.array([v.x_val, v.y_val, -v.z_val], dtype=float)

    def get_object_position(self, object_name: str) -> Optional[List[float]]:
        object_names = self.client.simListSceneObjects(object_name + ".*")
        if not object_names:
            return None
        pose = self.client.simGetObjectPose(object_names[0])
        return [pose.position.x_val, pose.position.y_val, -pose.position.z_val]

    def take_control(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)

    def release_control(self):
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)

    def disconnect(self):
        self.release_control()
