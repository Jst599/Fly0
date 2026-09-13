from typing import Dict, Any
import re
from ..ui.ui_utils import Colors


class CommandClassifier:

    # 明显的简单指令关键词，跳过 LLM 分类，秒级响应
    SIMPLE_PATTERNS = [
        # 中文 - 移动指令
        r'向[前后左右上下]飞?\s*\d+', r'往[前后左右上下]飞?\s*\d+',
        r'[前后]进\s*\d+', r'后[退撤]\s*\d+',
        r'起飞', r'降落', r'上升\s*\d+', r'下降\s*\d+', r'升高\s*\d+',
        r'左转\s*\d+', r'右转\s*\d+', r'转[向到]\s*\d+', r'旋转\s*\d+',
        r'悬停', r'回到起飞点', r'飞到高处', r'飞到低处',
        r'飞[高上]\s*\d+', r'飞[低下]\s*\d+',
        # 英文
        r'fly\s+(forward|backward|left|right|up|down)\s*\d+',
        r'take\s*off', r'land',
        r'turn\s+(left|right)\s*\d+',
        r'ascend', r'descend', r'hover',
    ]

    # 目标/地标/方位词已经足以判定这是视觉导航请求。
    # 不让 Qwen3-VL 额外做 visual/simple 分类，避免 Ollama 短暂 502
    # 阻塞本来应该直接进入 detector 的请求。
    VISUAL_PATTERNS = [
        r'(?:红|橙|黄|绿|蓝|黑|白|灰色?)?(?:球体?|方块|立方体|箱子|车|汽车|建筑|树|水塔|目标|物体)',
        r'(?:左|右|前|后|左前|右前|左侧|右侧|前方|后方).*(?:目标|物体|球|方块|建筑|车)',
        r'(?:飞向|前往|靠近|接近|寻找|找到|绕过|绕开)',
        r'\b(?:red ball|orange cube|target|object|car|building|tree|water tower)\b',
        r'\b(?:fly to|move to|go to|approach|find|locate|near)\b',
    ]

    def __init__(self, llm_interface):
        self.llm = llm_interface

    def _is_obviously_simple(self, command: str) -> bool:
        """关键词预判：明显是简单移动指令，不走 LLM"""
        cmd_lower = command.strip().lower()
        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True
        return False

    def classify(self, command: str) -> str:
        # 关键词预判 → 秒级响应
        if self._is_obviously_simple(command):
            return 'simple'

        cmd_lower = command.strip().lower()
        if any(re.search(pattern, cmd_lower) for pattern in self.VISUAL_PATTERNS):
            return 'visual'

        classification_prompt = f"""Determine if the following drone command requires visual detection (image recognition, target localization).
        Command: "{command}"
        Answer only one of the following:
        - visual: if command requires identifying targets in images (e.g., "fly to the car on the left", "fly to the red ball", "fly to the tree ahead")
        - simple: if command does not require visual detection (e.g., "fly up 10 meters", "takeoff", "land", "turn left 90 degrees")
        Answer only visual or simple, no other content."""
        
        try:
            temp_messages = [
                {"role": "system", "content": "You are a drone command classifier."},
                {"role": "user", "content": classification_prompt}
            ]
            response = self.llm.provider.chat(temp_messages, stream=False, temperature=0, max_tokens=10)
            
            if response:
                result = response.strip().lower()
                if result == 'visual':
                    return 'visual'
                elif result == 'simple':
                    return 'simple'
            
            return 'simple'
            
        except Exception as e:
            print(f"{Colors.RED}Classification failed: {e}{Colors.RESET}")
            return 'simple'
