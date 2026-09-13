from typing import Optional
import re
from ..ui.ui_utils import Colors, Banner


class CommandProcessor:
    
    def __init__(self, llm_interface, command_classifier, code_executor, navigation_handler):
        self.llm = llm_interface
        self.command_classifier = command_classifier
        self.code_executor = code_executor
        self.navigation_handler = navigation_handler
        self.mode = "flight"
    
    def process(self, command: str):
        command = command.strip()
        
        if not command:
            return
        
        if command == "!quit":
            print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
            import sys
            sys.exit(0)
        elif command == "!clear":
            self.llm.clear_history()
            print(f"{Colors.GREEN}✓ Chat history cleared{Colors.RESET}")
            return
        elif command == "!mode":
            modes = ["flight", "chat", "visual"]
            current_index = modes.index(self.mode)
            self.mode = modes[(current_index + 1) % len(modes)]
            print(f"{Colors.GREEN}✓ Mode switched to: {self.mode}{Colors.RESET}")
            return
        elif command == "!visual":
            self.mode = "visual"
            print(f"{Colors.GREEN}✓ Mode switched to: {self.mode}{Colors.RESET}")
            return
        elif command == "!help":
            Banner.show_help()
            return
        
        print(f"\n{Colors.CYAN}User:{Colors.RESET} {command}")

        # 确定性基础动作不经过 VLM。这样 Qwen3-VL 的 thinking 行为、
        # 上下文污染或代码围栏格式都不会阻塞起飞/降落等安全动作。
        if self._process_deterministic_command(command):
            return
        
        if self.mode == "visual":
            self.navigation_handler.navigate_to_target(command)
        else:
            command_type = self.command_classifier.classify(command)
            
            if command_type == 'visual' and self.mode == "flight":
                self.navigation_handler.navigate_to_target(command)
            else:
                self._process_llm_command(command)

    def _process_deterministic_command(self, command: str) -> bool:
        """处理不需要视觉或语言推理的基础飞行动作。"""
        c = command.strip().lower()
        try:
            if c in {"起飞", "起飞！", "takeoff", "take off"}:
                self.code_executor.drone.takeoff()
                print(f"{Colors.GREEN}✓ 已起飞{Colors.RESET}")
                return True
            number = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百点]+)"
            m = re.fullmatch(rf"起飞(?:到|至)?\s*({number})\s*(?:米|m)?(?:高|高度)?", c)
            if m:
                altitude = self._parse_number(m.group(1))
                self.code_executor.drone.takeoff()
                p = self.code_executor.drone.get_position()
                target_height = self.code_executor.drone.relative_altitude_target(altitude)
                reached = self.code_executor.drone.fly_to([p[0], p[1], target_height])
                relative = self.code_executor.drone.get_relative_altitude()
                reference = self.code_executor.drone.takeoff_reference_ned_z
                if reached:
                    print(f"{Colors.GREEN}✓ 已起飞并到达相对落地点高度 {relative:.2f} 米（目标 {altitude:.2f} 米，基准 NED Z={reference:.6f}）{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ 未到达相对落地点高度，当前 {relative:.2f} 米（目标 {altitude:.2f} 米）{Colors.RESET}")
                return True
            if c in {"降落", "降落！", "land"}:
                self.code_executor.drone.land()
                print(f"{Colors.GREEN}✓ 已降落{Colors.RESET}")
                return True
            if c in {"悬停", "hover"}:
                self.code_executor.drone.hover()
                print(f"{Colors.GREEN}✓ 已悬停{Colors.RESET}")
                return True

            if c in {"回到原点", "返回原点", "回到起飞点", "返回起飞点", "return home", "go home"}:
                # 这里的 [0, 0, 0] 使用 DroneController 的高度制坐标：
                # x/y 为世界原点，z=0 为地面高度。
                reached = self.code_executor.drone.fly_to([0.0, 0.0, 0.0])
                if reached:
                    print(f"{Colors.GREEN}✓ 已回到原点 [0, 0, 0]{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ 未到达原点，请检查实际位置与误差{Colors.RESET}")
                return True

            # 绝对坐标指令必须在视觉分类之前处理，否则“飞到[0,0,0]”
            # 会被通用的“飞到”关键词误判成目标搜索。
            m = re.fullmatch(
                r"(?:飞到|飞往|移动到|前往)\s*[\[（(【]\s*"
                r"(-?\d+(?:\.\d+)?)\s*[,，]\s*"
                r"(-?\d+(?:\.\d+)?)\s*[,，]\s*"
                r"(-?\d+(?:\.\d+)?)\s*[\]）)】]",
                c
            )
            if m:
                x, y, z = map(float, m.groups())
                reached = self.code_executor.drone.fly_to([x, y, z])
                if reached:
                    print(f"{Colors.GREEN}✓ 已飞到 [{x:.2f}, {y:.2f}, {z:.2f}]{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ 未到达 [{x:.2f}, {y:.2f}, {z:.2f}]{Colors.RESET}")
                return True

            # 这些动作沿用项目约定：世界坐标 X=前、Y=右、Z=高度。
            m = re.fullmatch(rf"(?:向|往)?上(?:飞|升)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(0, 0, self._parse_number(m.group(1)))
            m = re.fullmatch(rf"(?:向|往)?下(?:飞|降)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(0, 0, -self._parse_number(m.group(1)))
            m = re.fullmatch(rf"(?:向|往)?前(?:飞|进)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(self._parse_number(m.group(1)), 0, 0)
            m = re.fullmatch(rf"(?:向|往)?后(?:飞|退)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(-self._parse_number(m.group(1)), 0, 0)
            m = re.fullmatch(rf"(?:向|往)?左(?:飞)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(0, -self._parse_number(m.group(1)), 0)
            m = re.fullmatch(rf"(?:向|往)?右(?:飞)?\s*({number})\s*(?:米|m)?", c)
            if m:
                return self._move_relative(0, self._parse_number(m.group(1)), 0)
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ 基础动作执行失败: {e}{Colors.RESET}")
            return True

    @staticmethod
    def _parse_number(value: str) -> float:
        """Parse Arabic or common Chinese integer numerals in commands."""
        value = value.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            return float(value)
        if "点" in value:
            integer, fraction = value.split("点", 1)
            fraction_digits = {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2",
                               "三": "3", "四": "4", "五": "5", "六": "6", "七": "7",
                               "八": "8", "九": "9"}
            return CommandProcessor._parse_number(integer or "零") + float(
                "0." + "".join(fraction_digits[ch] for ch in fraction)
            )
        digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
                  "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if value in digits:
            return float(digits[value])
        total = 0
        section = 0
        for ch in value:
            if ch in digits:
                section = section * 10 + digits[ch]
            elif ch == "十":
                section = 10 if section == 0 else section * 10
            elif ch == "百":
                section = 100 if section == 0 else section * 100
            else:
                raise ValueError(f"无法解析数字: {value}")
            if ch == "百":
                total += section
                section = 0
            elif ch == "十":
                total += section
                section = 0
        return float(total + section)

    def _move_relative(self, dx: float, dy: float, dz: float) -> bool:
        drone = self.code_executor.drone
        p = drone.get_position()
        target = [p[0] + dx, p[1] + dy, p[2] + dz]
        reached = drone.fly_to(target)
        if reached:
            print(f"{Colors.GREEN}✓ 已移动到 [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ 未到达目标位置 [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]{Colors.RESET}")
        return True
    
    def _process_llm_command(self, command: str):
        """Process commands that go through LLM"""
        print(f"{Colors.CYAN}Assistant:{Colors.RESET}", end="", flush=True)
        
        if self.mode == "flight":
            code = self.llm.process(command, mode="flight")
            print()
            
            if code:
                print(f"{Colors.YELLOW}Generated code:{Colors.RESET}")
                print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")
                print(code)
                print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}\n")
                
                self.code_executor.execute(code)
            else:
                print(f"{Colors.RED}✗ No valid code detected{Colors.RESET}")
        else:
            response = self.llm.process(command, mode="chat")
            print()
