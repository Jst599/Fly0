from typing import Tuple
import numpy as np
from ..ui.ui_utils import Colors


class NavigationHandler:
    def __init__(self, planner):
        self.planner = planner
        # Visual grounding returns the visible surface point.  That point is
        # not a safe vehicle goal, so stop short of it by this distance.
        self.target_standoff = 3.0
        self.min_target_standoff = 2.0
        self.overhead_clearance = 5.0

    def _safe_approach_point(self, target_position) -> Tuple[np.ndarray, float, float]:
        """Return an AirSim-world NED waypoint in front of the observed target.

        ``VisualTargetDetector.calculate_target_position`` returns a world NED
        point.  The point is usually on the object's visible surface; moving
        directly to it can place the vehicle inside the collision volume.  We
        therefore retreat along the current-drone -> target ray while keeping
        the same coordinate convention as EgoPlanner.
        """
        target = np.asarray(target_position, dtype=float)
        current = np.asarray(self.planner.get_drone_pose()["position"], dtype=float)
        ray = target - current
        distance = float(np.linalg.norm(ray))
        if not np.isfinite(distance) or distance < 1e-3:
            raise ValueError("target point is coincident with the drone or invalid")

        # Never retreat behind the current vehicle when the target is already
        # closer than the desired clearance.
        desired_standoff = max(self.min_target_standoff, distance * 0.5)
        standoff = min(distance, self.target_standoff, desired_standoff)
        approach = target - ray / distance * standoff
        return approach, distance, standoff

    def _navigation_goal(self, target_position, user_command: str):
        """Build a command-aware AirSim NED goal from a detected surface point."""
        target = np.asarray(target_position, dtype=float)
        current = np.asarray(self.planner.get_drone_pose()["position"], dtype=float)
        distance = float(np.linalg.norm(target - current))
        if "上方" in user_command or "顶部" in user_command or "顶上" in user_command:
            # AirSim uses NED coordinates, therefore a smaller Z is higher.
            goal = target.copy()
            goal[2] = target[2] - self.overhead_clearance
            return goal, distance, self.overhead_clearance, "overhead"
        goal, distance, standoff = self._safe_approach_point(target)
        return goal, distance, standoff, "standoff"
    
    def navigate_to_target(self, user_command: str) -> bool:
        print(f"{Colors.YELLOW}Navigating with EgoPlanner...{Colors.RESET}")

        try:
            # 先停止上一个还在跑的规划线程，防止 IOLoop 冲突
            self.planner.stop_planning()

            client = self.planner.client
            detector = self.planner.visual_detector
            
            # print(f"{Colors.CYAN}[DEBUG] Visual detector: {detector}{Colors.RESET}")
            
            if not detector:
                print(f"{Colors.RED}✗ Visual detector not enabled{Colors.RESET}")
                return False
            
            # print(f"{Colors.CYAN}[DEBUG] Starting visual target detection for command: {user_command}{Colors.RESET}")
            success, target_position = detector.detect_visual_target(client, user_command, search_360=True)
            
            # print(f"{Colors.CYAN}[DEBUG] Detection success: {success}, target_position: {target_position}{Colors.RESET}")
            
            if success and target_position != (0, 0, 0):
                print(f"{Colors.GREEN}✓ Target surface (AirSim world NED): X={target_position[0]:.2f}, Y={target_position[1]:.2f}, Z={target_position[2]:.2f}{Colors.RESET}")

                goal_position, target_distance, clearance, goal_kind = self._navigation_goal(
                    target_position, user_command
                )
                if goal_kind == "overhead":
                    print(
                        f"{Colors.CYAN}Overhead waypoint (AirSim world NED): "
                        f"X={goal_position[0]:.2f}, Y={goal_position[1]:.2f}, Z={goal_position[2]:.2f}; "
                        f"surface distance={target_distance:.2f}m, vertical clearance={clearance:.2f}m{Colors.RESET}"
                    )
                else:
                    print(
                        f"{Colors.CYAN}Approach waypoint (AirSim world NED): "
                        f"X={goal_position[0]:.2f}, Y={goal_position[1]:.2f}, Z={goal_position[2]:.2f}; "
                        f"surface distance={target_distance:.2f}m, standoff={clearance:.2f}m{Colors.RESET}"
                    )
                
                print(f"{Colors.YELLOW}Starting path planning and navigation...{Colors.RESET}")
                self.planner.take_control()
                max_replans = 2
                for attempt in range(max_replans + 1):
                    self.planner.plan_to_position(goal_position)
                    status, reason = self.planner.wait_for_result(timeout=180.0)
                    if status == "success":
                        print(f"Navigation complete ({reason})")
                        return True
                    if status != "collision_escape":
                        print(f"Navigation failed: {status} ({reason})")
                        return False
                    if attempt >= max_replans:
                        print("Navigation aborted after collision; replan limit reached")
                        return False
                    goal_position = np.asarray(goal_position, dtype=float).copy()
                    goal_position[2] -= 5.0
                    print(f"Collision escape complete; replanning attempt {attempt + 1}")
                return False
                if False and status == "success":
                    print(f"{Colors.GREEN}✓ Navigation complete ({reason}){Colors.RESET}")
                    return True
                if status == "collision_escape":
                    print(f"{Colors.RED}✗ Navigation aborted after collision; emergency escape executed{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ Navigation failed: {status} ({reason}){Colors.RESET}")
                return False
            else:
                print(f"{Colors.RED}✗ Failed to get valid target position{Colors.RESET}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}✗ Navigation error: {str(e)}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False
