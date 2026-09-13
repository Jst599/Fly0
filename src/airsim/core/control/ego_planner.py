import airsim
import numpy as np
import math
import time
from scipy.spatial import KDTree
import threading
from queue import Queue
from ..detection.visual_target_detector import VisualTargetDetector


draw=True
class EgoPlanner:
    def __init__(self, drone_name="Drone1", lidar_sensors=["LidarSensor1"], visual_detector=None):
        # 初始化AirSim客户端
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        
        self.drone_name = drone_name
        self.lidar_sensors = lidar_sensors
        self.visual_detector = visual_detector
        self.max_velocity = 2.0  # 最大速度 m/s
        self.max_acceleration = 2.0  # 最大加速度 m/s²
        self.horizon_distance = 10.0  # 规划视距
        self.drone_radius = 1.0  # 无人机半径（用于碰撞检测）
        self.goal_dis = 0.0
        self.lidar_range = 30.0  # 雷达探测范围
        self.safety_distance = 1.0  # 安全距离
        self.collision_check_resolution = 0.5  # 碰撞检测分辨率
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.goal_position = np.zeros(3)
        self.is_running = False
        self.obstacles = []
        self.obstacle_kdtree = None
        
        # B样条曲线参数
        self.control_points = []  # 控制点
        self.bspline_degree = 3   # B样条曲线阶数
        
        # 历史航线点
        self.trajectory_history = []  # 存储历史轨迹点
        
        # 路径规划线程
        self.planning_thread = None
        self.command_queue = Queue()
        self._lock = threading.Lock()  # AirSim 客户端线程锁
        self._collision_count_baseline = 0
        self._collision_timestamp_baseline = 0.0
        self._collision_signature_seen = None
        self.result_status = "idle"
        self.result_reason = ""

    def _call_airsim(self, func, *args, **kwargs):
        """线程安全的 AirSim 调用包装器"""
        with self._lock:
            return func(*args, **kwargs)
    
    def take_control(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
    
    
    def get_drone_pose(self):
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        kinematics = drone_state.kinematics_estimated
        position = kinematics.position
        orientation = kinematics.orientation
        linear_velocity = kinematics.linear_velocity
        angular_velocity = kinematics.angular_velocity
        return {
            'position': np.array([position.x_val, position.y_val, position.z_val]),
            'orientation': np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val]),
            'linear_velocity': np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val]),
            'angular_velocity': np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])
        }
    
    def get_drone_state(self):
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        return {
            'gps_location': drone_state.gps_location,
            'timestamp': drone_state.timestamp,
            'landed_state': drone_state.landed_state,
            'rc_data': drone_state.rc_data,
            'ready': drone_state.ready,
            'ready_message': drone_state.ready_message
        }
    
    def get_lidar_data(self, sensor_name):
        try:
            lidar_data = self.client.getLidarData(sensor_name, self.drone_name)

            if len(lidar_data.point_cloud) < 3:
                return None
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            return {
                'points': points,
                'time_stamp': lidar_data.time_stamp,
                'pose': lidar_data.pose
            }
        except Exception as e:
            return None
    
    def euler_to_rotation_matrix(self, euler_angles):
        roll, pitch, yaw = euler_angles
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        return np.dot(Rz, np.dot(Ry, Rx))

    def get_all_lidar_data(self):
        all_points = []
        drone_pose = self.get_drone_pose()
        drone_position = drone_pose['position']
        drone_orientation = drone_pose['orientation']
        drone_rotation_matrix = self.quaternion_to_rotation_matrix(drone_orientation)
        for sensor in self.lidar_sensors:
            data = self.get_lidar_data(sensor)
            if data and len(data['points']) > 0:
                lidar_position = np.array([0, 0, -1.0])
                lidar_rotation = np.array([0, 0, 0]) 
                lidar_rotation_matrix = self.euler_to_rotation_matrix(lidar_rotation)
                body_points = np.dot(data['points'], lidar_rotation_matrix.T) + lidar_position
                world_points = np.dot(body_points, drone_rotation_matrix.T) + drone_position
                all_points.append(world_points)
        
        if all_points:
            ppp = np.vstack(all_points) 
            if draw:
                draw_point = []
                try:
                    for points in ppp:
                        draw_point.append(airsim.Vector3r(points[0], points[1], points[2]))
                    self.client.simPlotPoints(
                            draw_point, 
                            color_rgba=[0.0, 0.0, 1.0, 1.0], # 雷达点蓝色
                            size=5, 
                            duration=0.1, 
                            is_persistent=False)
                except Exception as e:
                    print(f"Error drawing debug points: {e}")
            return np.vstack(all_points)
        return np.array([]).reshape(0, 3)
    
    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def update_obstacles(self):
        lidar_points = self.get_all_lidar_data()
        if len(lidar_points) > 0:
            self.obstacles = lidar_points
            self.obstacle_kdtree = KDTree(self.obstacles)
    
    def check_collision(self, point):
        if self.obstacle_kdtree is None or len(self.obstacles) == 0:
            return False
        distance, _ = self.obstacle_kdtree.query(point)
        return distance < self.safety_distance
    
    def check_segment_collision(self, start, end):
        if self.obstacle_kdtree is None or len(self.obstacles) == 0:
            return False
        
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return False
        
        direction = direction / length
        num_samples = max(2, int(length / self.collision_check_resolution))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start + t * direction * length
            if self.check_collision(point):
                return True
        
        return False

    def get_collision_info(self):
        """读取 AirSim 碰撞状态。EgoPlanner 内部使用 NED 坐标。"""
        try:
            return self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        except Exception:
            return None

    def has_collision(self):
        info = self.get_collision_info()
        if info is None:
            return False
        # AirSim's Python CollisionInfo has no collision_count field.  The
        # has_collided flag is sticky and may describe an old ground contact,
        # so only a new timestamp after navigation began is an event.
        if not bool(getattr(info, "has_collided", False)):
            return False
        timestamp = float(getattr(info, "time_stamp", 0.0) or 0.0)
        object_name = str(getattr(info, "object_name", "") or "")
        if timestamp <= float(self._collision_timestamp_baseline):
            return False
        signature = (timestamp, object_name)
        previous = self._collision_signature_seen
        if previous is not None and previous[1] == object_name:
            penetration = float(getattr(info, "penetration_depth", 0.0) or 0.0)
            print(f"Confirmed collision: object={object_name or '<unknown>'}, timestamp={timestamp:.0f}, penetration={penetration:.3f}m")
            return True
        # Require the same new event to be observed twice, filtering one-frame
        # simulator/contact jitter without delaying a genuine collision.
        self._collision_signature_seen = signature
        return False

    def emergency_escape(self, climb_meters=30.0, climb_speed=3.0):
        """碰撞后立即取消旧任务、制动并向上脱困。

        AirSim 使用 NED：向上是 z 减小，因此这里使用 vz=-climb_speed。
        30 米是脱困上限；如果场景有低天花板，应在配置中调小。
        """
        print(f"⚠ 检测到碰撞，取消当前路径并紧急上升 {climb_meters:.1f}m")
        try:
            self.client.cancelLastTask()
        except Exception:
            pass
        time.sleep(0.2)
        try:
            # 先给一个短暂零速度指令，降低继续顶住物体的时间。
            self.client.moveByVelocityAsync(
                0.0, 0.0, 0.0, 0.4,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=self.drone_name
            ).join()
            # NED 的负 Z 是向上；3m/s × 10s = 30m。
            self.client.moveByVelocityAsync(
                0.0, 0.0, -abs(climb_speed), climb_meters / abs(climb_speed),
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=self.drone_name
            ).join()
            self.client.hoverAsync(vehicle_name=self.drone_name).join()
        except Exception as exc:
            print(f"⚠ 紧急脱困指令失败: {exc}")
        finally:
            # 当前目标已不再可信，必须结束本次导航，避免恢复后反复撞击。
            self.is_running = False
            self.result_status = "collision_escape"
            self.result_reason = f"collision; emergency climb {climb_meters:.1f}m"
    
    def find_consecutive_colliding_segments(self, control_points):
        colliding_segments = []
        
        for i in range(len(control_points) - 1):
            if self.check_segment_collision(control_points[i], control_points[i+1]):
                start_idx = i
                end_idx = i + 1
                

                while start_idx > 0 and self.check_segment_collision(control_points[start_idx-1], control_points[start_idx]):
                    start_idx -= 1
                
                while end_idx < len(control_points) - 1 and self.check_segment_collision(control_points[end_idx], control_points[end_idx+1]):
                    end_idx += 1
                
                segment = (start_idx, end_idx)
                if not colliding_segments or segment != colliding_segments[-1]:
                    colliding_segments.append(segment)
        
        return colliding_segments
    
    def path_search(self, segment_indices, avoid_distance=4.0):
        start_idx, end_idx = segment_indices
        start_point = self.control_points[start_idx]
        end_point = self.control_points[end_idx]
        mid_point = (start_point + end_point) / 2.0

        if len(self.obstacles) == 0:
            return [start_point, end_point]

        obstacle_center = np.mean(self.obstacles, axis=0)
        base_avoid_dir = mid_point - obstacle_center
        base_dir_norm = np.linalg.norm(base_avoid_dir)
        if base_dir_norm < 0.01:
            base_avoid_dir = np.array([1.0, 0.0, 0.0])
        else:
            base_avoid_dir = base_avoid_dir / base_dir_norm

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(base_avoid_dir, up)
        if np.linalg.norm(right) < 0.01:
            right = np.array([0.0, 1.0, 0.0])
        right = right / np.linalg.norm(right)

        candidates = [
            base_avoid_dir,
            (base_avoid_dir + up * 0.5),
            (base_avoid_dir - up * 0.5),
            (base_avoid_dir + right * 0.5),
            (base_avoid_dir - right * 0.5),
            up, (up + right * 0.5), (up - right * 0.5),
        ]
        candidates = [d / np.linalg.norm(d) for d in candidates]

        best_path = None
        best_score = float("-inf")
        for direction in candidates:
            for dist in [avoid_distance, avoid_distance * 1.5]:
                avoid_point = start_point + direction * dist
                safe_a = not self.check_segment_collision(start_point, avoid_point)
                safe_b = not self.check_segment_collision(avoid_point, end_point)
                if safe_a and safe_b:
                    score = -np.linalg.norm(avoid_point - end_point)
                    if score > best_score:
                        best_score = score
                        best_path = [start_point, avoid_point, end_point]

        if best_path is not None:
            if draw:
                mid_avoid = best_path[1]
                draw_pt = [airsim.Vector3r(mid_avoid[0], mid_avoid[1], mid_avoid[2])]
                self.client.simPlotPoints(
                    draw_pt, color_rgba=[1.0, 0.65, 0.0, 1.0],
                    size=25, duration=5.0, is_persistent=False)
            return best_path

        # 全部方向碰撞 → 检查是否因为起点离障碍物太近
        nearest_dist = np.min(np.linalg.norm(self.obstacles - start_point, axis=1)) if len(self.obstacles) > 0 else float('inf')
        if nearest_dist < self.safety_distance * 2:
            # This is a pre-flight danger condition, not a valid route. Do
            # one emergency escape and terminate the navigation attempt;
            # returning a short waypoint used to make the planner recompute
            # the same colliding segment until it timed out.
            print(f"⚠ 当前位置离障碍物仅 {nearest_dist:.1f}m，终止路径并紧急上升！")
            # A LiDAR/planning failure is not proof of a physical collision.
            # Only the explicit has_collision() branches may trigger the
            # emergency climb.  Stop this navigation attempt in place.
            self.is_running = False
            self.result_status = "blocked"
            self.result_reason = f"no collision-free detour; nearest obstacle {nearest_dist:.2f}m"
            return [start_point]
        else:
            safe_point = start_point + np.array([0, 0, 3.0])
            return [start_point, safe_point, end_point]

    def find_pv_pairs(self, control_point, path):
        """寻找p-v对（控制点和排斥方向向量）control_point：p of {p, v}, path：[start_point, avoid_point, end_point]"""
        pv_pairs = []
        
        # 找到路径上离控制点最近的点
        min_dist = float('inf')
        closest_point = None
        
        # path：[start_point, avoid_point, end_point]
        for i in range(len(path) - 1):
            segment_start = path[i]
            segment_end = path[i+1]
            
            # 计算控制点到线段的最近点
            segment_vec = segment_end - segment_start
            segment_len = np.linalg.norm(segment_vec)
            
            if segment_len == 0:
                continue
                
            segment_dir = segment_vec / segment_len
            t = np.dot(control_point - segment_start, segment_dir) / segment_len
            t = max(0, min(1, t))  # 限制在0到1之间
            
            closest_on_segment = segment_start + t * segment_vec
            dist = np.linalg.norm(control_point - closest_on_segment)
            
            if dist < min_dist and dist > 0:
                min_dist = dist
                closest_point = closest_on_segment
        
        if closest_point is not None and min_dist < self.safety_distance * 2:
            # 计算排斥方向（从控制点指向路径上的最近点）
            repulsive_dir = control_point - closest_point
            repulsive_dir_norm = np.linalg.norm(repulsive_dir)
            
            if repulsive_dir_norm > 0:
                repulsive_dir = repulsive_dir / repulsive_dir_norm
                pv_pairs.append((closest_point, repulsive_dir))

            if draw:
                # ② 绘制排斥方向箭头
                try:
                    direction_vector =  closest_point - control_point 
                    arrow_end = control_point + direction_vector * repulsive_dir
                    start_vec = airsim.Vector3r(control_point[0], control_point[1], control_point[2])
                    end_vec = airsim.Vector3r(arrow_end[0], arrow_end[1], arrow_end[2])
                    
                    # 绘制红色箭头，持续时间0.1秒
                    self.client.simPlotArrows(
                        [start_vec], 
                        [end_vec], 
                        color_rgba=[1.0, 0.0, 0.0, 1.0], 
                        thickness=5.0, 
                        arrow_size=10.0, 
                        duration=0.1, 
                        is_persistent=False
                    )
                except Exception as e:
                    print(f"绘制排斥方向箭头失败: {e}")
                
            
        return pv_pairs
    
    def optimize_trajectory_with_pv(self, control_points, pv_pairs_list):
        """使用p-v对优化轨迹"""
        if not pv_pairs_list or len(pv_pairs_list) != len(control_points):
            return control_points
        
        optimized_points = np.copy(control_points)
        repulsion_strength = 7.0  # 排斥强度
        for i in range(len(control_points)):
            if pv_pairs_list[i]:  # 如果有p-v对
                for p, v in pv_pairs_list[i]:
                    # 应用排斥力
                    optimized_points[i] = optimized_points[i] - v * repulsion_strength
        return optimized_points
    
    def generate_bspline_control_points(self, start_pos, goal_pos, num_points=10):
        """生成B样条曲线的控制点"""
        # 简化的控制点生成，实际应用中应该使用更复杂的算法
        control_points = [start_pos]
        if self.goal_dis <= 15.0:
            num_points = 5
        if self.goal_dis <= 10.0:
            num_points = 2
        for i in range(1, num_points - 1):
            alpha = i / (num_points - 1)
            point = start_pos * (1 - alpha) + goal_pos * alpha
            
            control_points.append(point)
        control_points.append(goal_pos)
        if draw:
            draw_points = []
            for points in control_points:
                draw_points.append(airsim.Vector3r(points[0], points[1], points[2]))
            try:
                self.client.simPlotPoints(
                    draw_points, 
                    color_rgba=[0.5, 0.5, 0.5, 1.0], # 初始B样条灰色
                    size=20, 
                    duration=0.1, 
                    is_persistent=False
                )
            except Exception as e:
                print(f"Error drawing debug points: {e}")
        
        return np.array(control_points)
    
    def bspline_interpolate(self, control_points, num_samples=100):
        """B样条曲线插值 - 使用scipy实现，稳定可靠"""
        if len(control_points) < 2:
            return np.array(control_points)

        # 少于4个点时用线性插值（scipy splprep 至少需要 4 个点）
        ctrl = np.asarray(control_points)
        if len(ctrl) < 4:
            t_orig = np.linspace(0, 1, len(ctrl))
            t_new = np.linspace(0, 1, num_samples)
            result = np.zeros((num_samples, ctrl.shape[1]))
            for dim in range(ctrl.shape[1]):
                result[:, dim] = np.interp(t_new, t_orig, ctrl[:, dim])
            return result

        # 用 scipy 的三次 B 样条
        try:
            from scipy.interpolate import splprep, splev
            # transpose to (dims, N) format required by splprep
            pts = ctrl.T
            k = min(3, len(ctrl) - 1)  # 三次样条，点数不够就降阶
            tck, u = splprep(pts, s=0, k=k)
            u_new = np.linspace(0, 1, num_samples)
            trajectory = np.array(splev(u_new, tck)).T
            return trajectory
        except Exception:
            # 兜底：线性插值
            t_orig = np.linspace(0, 1, len(ctrl))
            t_new = np.linspace(0, 1, num_samples)
            result = np.zeros((num_samples, ctrl.shape[1]))
            for dim in range(ctrl.shape[1]):
                result[:, dim] = np.interp(t_new, t_orig, ctrl[:, dim])
            return result
    
    def safe_move_to_position(self, x, y, z, velocity, min_distance=0.5):
        """安全移动到指定位置"""
        # 获取当前位置
        pose = self.get_drone_pose()
        current_pos = pose['position']
        
        # 计算距离
        target_pos = np.array([x, y, z])
        distance = np.linalg.norm(target_pos - current_pos)
        
        # 如果距离过小，使用速度控制
        if distance < min_distance:
            if distance < 1:  # 非常接近目标
                self.client.hoverAsync(vehicle_name=self.drone_name)
                return
            
            # 计算方向向量
            direction = target_pos - current_pos
            direction = direction / np.linalg.norm(direction)
            
            # 使用速度控制移动短距离
            move_time = distance / velocity
            
            self.client.moveByVelocityAsync(
                direction[0] * velocity,
                direction[1] * velocity,
                direction[2] * velocity,
                move_time,
                vehicle_name=self.drone_name,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0)
            )
        else:
            # 正常使用位置控制
            self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.drone_name, 
                                            )

    def planning_loop(self):
        """Ego-Planner 主循环，碰撞/卡死时立即终止当前任务。"""
        max_iterations = 20
        max_same_segments = 3
        iteration = 0
        prev_segments = None
        same_count = 0
        try:
            while self.is_running and iteration < max_iterations:
                iteration += 1
                try:
                    pose = self.get_drone_pose()
                    self.current_position = pose['position']
                    self.current_velocity = pose['linear_velocity']
                    self.trajectory_history.append(self.current_position.copy())
                    if len(self.trajectory_history) > 500:
                        self.trajectory_history = self.trajectory_history[-400:]

                    if self.has_collision():
                        self.emergency_escape()
                        break

                    distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
                    self.goal_dis = distance_to_goal
                    if distance_to_goal < 1.0:
                        try:
                            self.client.cancelLastTask()
                            self.client.hoverAsync(vehicle_name=self.drone_name).join()
                        except Exception:
                            pass
                        self.is_running = False
                        self.result_status = "success"
                        self.result_reason = f"final error {distance_to_goal:.2f}m"
                        print(f"✓ 已到达目标位置，最终误差 {distance_to_goal:.2f}m")
                        break

                    self.update_obstacles()
                    self.control_points = self.generate_bspline_control_points(
                        self.current_position, self.goal_position)
                    colliding_segments = self.find_consecutive_colliding_segments(self.control_points)

                    if colliding_segments:
                        seg_key = str(colliding_segments)
                        if seg_key == prev_segments:
                            same_count += 1
                        else:
                            same_count = 1
                            prev_segments = seg_key
                        if same_count > max_same_segments:
                            print("⚠ 同一碰撞段反复出现，立即执行紧急脱困")
                            try:
                                self.client.cancelLastTask()
                            except Exception:
                                pass
                            self.is_running = False
                            self.result_status = "blocked"
                            self.result_reason = "repeated LiDAR collision segment without physical collision"
                            break

                        print(f"检测到碰撞段: {colliding_segments}，正在优化路径...")
                        pv_pairs_list = [[] for _ in range(len(self.control_points))]
                        for segment in colliding_segments:
                            path = self.path_search(segment)
                            if self.result_status != "running":
                                break
                            for j in range(segment[0], segment[1] + 1):
                                if j < len(self.control_points):
                                    pv_pairs_list[j].extend(self.find_pv_pairs(self.control_points[j], path))
                        if self.result_status != "running":
                            break
                        control_points = self.optimize_trajectory_with_pv(self.control_points, pv_pairs_list)
                        trajectory = self.bspline_interpolate(control_points, num_samples=50)
                    else:
                        trajectory = self.bspline_interpolate(self.control_points, num_samples=30)

                    if trajectory is None or len(trajectory) <= 1:
                        continue
                    step = max(1, len(trajectory) // 20)
                    waypoints = trajectory[::step]
                    if len(waypoints) > 1 and not np.array_equal(waypoints[-1], trajectory[-1]):
                        waypoints = np.vstack([waypoints, trajectory[-1:]])
                    airsim_points = [airsim.Vector3r(p[0], p[1], p[2]) for p in waypoints]

                    try:
                        self.client.moveOnPathAsync(
                            airsim_points, velocity=self.max_velocity, timeout_sec=60,
                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                            yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.drone_name)
                        deadline = time.time() + 60.0
                        last_pos = self.current_position.copy()
                        stagnant_since = None
                        while self.is_running and time.time() < deadline:
                            if self.has_collision():
                                self.emergency_escape()
                                break
                            pose_now = self.get_drone_pose()
                            moved = np.linalg.norm(pose_now['position'] - last_pos)
                            speed = np.linalg.norm(pose_now['linear_velocity'])
                            if moved < 0.15 and speed < 0.20:
                                if stagnant_since is None:
                                    stagnant_since = time.time()
                                elif (time.time() - stagnant_since >= 4.0 and
                                      np.linalg.norm(pose_now['position'] - self.goal_position) > 1.0):
                                    print("⚠ 无人机疑似被卡住（4秒无有效位移），立即执行紧急脱困")
                                    # A stalled pose without a collision event
                                    # is not sufficient evidence of contact.
                                    # Cancel this path and report failure; keep
                                    # the emergency climb exclusively for the
                                    # explicit has_collision() branches.
                                    try:
                                        self.client.cancelLastTask()
                                    except Exception:
                                        pass
                                    self.is_running = False
                                    self.result_status = "timeout"
                                    self.result_reason = "no progress without collision"
                                    break
                            else:
                                stagnant_since = None
                                last_pos = pose_now['position'].copy()
                            if np.linalg.norm(pose_now['position'] - waypoints[-1]) < 1.0:
                                break
                            time.sleep(0.1)
                        if self.is_running and time.time() >= deadline:
                            print("⚠ 路径执行超时，取消当前路径")
                            self.client.cancelLastTask()
                            self.is_running = False
                            self.result_status = "timeout"
                            self.result_reason = "path timeout"
                    except Exception as exc:
                        print(f"路径执行中断: {exc}")
                        self.is_running = False
                        self.result_status = "error"
                        self.result_reason = str(exc)
                except Exception as exc:
                    print(f"规划循环异常: {exc}")
                    self.is_running = False
                    self.result_status = "error"
                    self.result_reason = str(exc)
            if self.is_running:
                self.is_running = False
                self.result_status = "timeout"
                self.result_reason = f"max iterations {max_iterations}"
                print("⚠ 导航未完成：达到最大规划次数")
        except Exception as exc:
            self.is_running = False
            self.result_status = "error"
            self.result_reason = str(exc)
            print(f"规划线程异常: {exc}")
    
    def plan_to_position(self, goal_position):
        """规划到指定位置"""
        self.goal_position = np.array(goal_position)
        self.result_status = "running"
        self.result_reason = ""
        info = self.get_collision_info()
        self._collision_count_baseline = int(getattr(info, "collision_count", 0) or 0)
        self._collision_timestamp_baseline = float(
            getattr(info, "time_stamp", 0.0) or 0.0
        ) if info is not None else 0.0
        self._collision_signature_seen = None
        self.is_running = True
        
        # 重置历史航线，从当前位置开始记录
        pose = self.get_drone_pose()
        self.trajectory_history = [pose['position'].copy()]
        
        # 启动规划线程
        self.planning_thread = threading.Thread(target=self.planning_loop)
        self.planning_thread.daemon = True
        self.planning_thread.start()

    def wait_for_result(self, timeout=180.0):
        """等待规划线程真实结束，返回 (status, reason)。"""
        thread = self.planning_thread
        if thread is not None:
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            self.stop_planning()
            self.result_status = "timeout"
            self.result_reason = f"wait timeout {timeout:.0f}s"
        return self.result_status, self.result_reason
    
    def stop_planning(self):
        """停止规划"""
        self.is_running = False
        if self.planning_thread and self.planning_thread.is_alive():
            self.planning_thread.join(timeout=3.0)
        # 不调 hoverAsync —— 可能在 msgpackrpc 里死锁
    
    
  
      





