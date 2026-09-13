import json
import re
import requests
import time
from typing import Optional, List
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        pass
    
    @abstractmethod
    def check_service(self) -> bool:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
    
    def check_service(self) -> bool:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Ollama service returned HTTP {response.status_code}: {response.text[:200]}")
                return False
            try:
                models = [m.get("name", "") for m in response.json().get("models", [])]
                if models and self.model not in models:
                    print(f"Ollama is reachable, but model '{self.model}' is not installed.")
                    print(f"Installed models: {', '.join(models)}")
                    print(f"Run: ollama pull {self.model}")
                    return False
            except (TypeError, ValueError, AttributeError):
                pass
            # /api/tags 只能说明服务在线；/api/show 能提前暴露模型 tag、
            # 模型格式或后端加载问题，避免第一次导航请求才收到 502。
            show = requests.post(
                f"{self.url}/api/show", json={"name": self.model}, timeout=15
            )
            if show.status_code != 200:
                print(f"Ollama cannot load model '{self.model}': HTTP {show.status_code} {show.text[:300]}")
                return False
            return True
        except requests.RequestException as exc:
            print(f"Ollama service unavailable at {self.url}: {exc}")
            return False
    
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        # Qwen3 系列默认可能返回 thinking 而 content 为空。Fly0 的控制链
        # 需要短、可执行的答案，因此显式关闭 thinking，并将 OpenAI 风格的
        # max_tokens 转为 Ollama 的 options.num_predict。
        options = dict(kwargs.pop("options", None) or {})
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            options.setdefault("temperature", temp)
        options.setdefault("temperature", 0)
        if "max_tokens" in kwargs:
            options.setdefault("num_predict", kwargs.pop("max_tokens"))
        options.setdefault("num_predict", 256)

        data = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": options
        }
        data.update(kwargs)
        # 不允许调用方意外重新打开思考模式。
        data["think"] = False

        response = None
        for attempt in range(3):
            response = requests.post(
                f"{self.url}/api/chat",
                json=data,
                stream=stream,
                timeout=(10, 60)
            )
            if response.status_code in (502, 503, 504) and attempt < 2:
                # Ollama may briefly return 502 while loading a multimodal model.
                time.sleep(2)
                continue
            if not stream and response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "") or ""
                if not content and result.get("done_reason") == "length" and attempt < 2:
                    data["options"]["num_predict"] = max(
                        1024, int(data["options"].get("num_predict", 512)) * 2
                    )
                    print(
                        "LLM output was truncated by thinking; retrying with "
                        f"num_predict={data['options']['num_predict']}"
                    )
                    continue
            break
        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Ollama /api/chat HTTP {response.status_code} for model '{self.model}': {detail}"
            )
        
        if stream:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_data = json.loads(chunk)
                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"]
                            response_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
            return response_text
        else:
            result = response.json()
            message = result.get("message", {}) or {}
            content = message.get("content", "") or ""
            if not content:
                # Keep the diagnostic useful without feeding an empty assistant
                # turn back into the next command's conversation history.
                reason = result.get("done_reason", "unknown")
                print(f"LLM returned empty content (done_reason={reason})")
            return content
    
    def chat_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Chat with image using Ollama API"""
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "images": [image_base64],
            "stream": stream,
            "think": False,
            "options": {"temperature": 0, "num_predict": 128}
        }

        response = None
        for attempt in range(3):
            response = requests.post(
                f"{self.url}/api/chat",
                json=data,
                stream=stream,
                timeout=120
            )
            if response.status_code not in (502, 503, 504) or attempt == 2:
                break
            time.sleep(2)
        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Ollama /api/chat HTTP {response.status_code} for model '{self.model}': {detail}"
            )
        
        if stream:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_data = json.loads(chunk)
                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"]
                            response_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
            return response_text
        else:
            result = response.json()
            return result.get("message", {}).get("content", "")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def check_service(self) -> bool:
        return self.client is not None
    
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            stream=stream
        )
        
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)
            print()
            return response
        else:
            return completion.choices[0].message.content
    
    def chat_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Chat with image using OpenAI-compatible API"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=20,
            temperature=0,
            stream=stream
        )
        
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)
            print()
            return response
        else:
            return completion.choices[0].message.content


class LLMInterface:
    
    def __init__(self, sys_prompt_path: str, config):
        self.config = config
        self.provider = self._create_provider()
        
        with open(sys_prompt_path, 'r', encoding='utf-8') as f:
            self.base_system_prompt = f.read()
        
        self.chat_history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": "向上飞10米"},
            {"role": "assistant", "content": 
             """
                ```python
                drone.fly_to([drone.get_position()[0], drone.get_position()[1], drone.get_position()[2] + 10])
                ```
                This code uses the fly_to() function to move the drone 10 units higher than its current position.
                It retrieves the current position using get_position(), then creates a new position with the same X and Y coordinates but the Z coordinate increased by 10.
                The drone then flies to this new position using fly_to().
             """
            }
        ]
    
    def _create_provider(self) -> LLMProvider:
        """Create LLM provider based on configuration"""
        api_type = self.config.get("API_TYPE", "ollama")
        
        if api_type == "ollama":
            control_config = self.config.get_control_config()
            # 视觉与文本控制统一使用配置中的 Qwen3-VL 模型。
            return OllamaProvider(
                control_config.get("model") or self.config.get("OLLAMA_MODEL", "qwen3-vl:4b"),
                control_config.get("base_url", "http://localhost:11434")
            )
        elif api_type == "openai":
            return OpenAIProvider(
                self.config.get("OPENAI_API_KEY", ""),
                self.config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                self.config.get("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
            )
        elif api_type == "vllm":
            return OpenAIProvider(
                self.config.get("OPENAI_API_KEY", "EMPTY"),
                self.config.get("VLLM_BASE_URL", "http://192.168.10.4:8000/v1"),
                self.config.get("VLLM_MODEL", "qwen2.5-vl-72b-instruct")
            )
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def _get_system_prompt(self, mode: str = "flight") -> str:
       
        return self.base_system_prompt
    
    def ask(self, prompt: str, mode: str = "flight") -> str:
        self.chat_history[0]["content"] = self._get_system_prompt(mode)
        self.chat_history.append({"role": "user", "content": prompt})
        
        if not self.provider.check_service():
            print("LLM service unavailable")
            return ""
        
        try:
            # Control commands need the complete response so code extraction
            # can see the final fenced block.  Qwen3/Ollama may emit thinking
            # or empty streamed content even when a non-stream response has
            # valid message.content.
            response = self.provider.chat(self.chat_history, stream=False, max_tokens=512)
            if response:
                print(response, end="", flush=True)
        except Exception as exc:
            print(f"LLM request failed: {exc}")
            return ""
        
        if response:
            self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def ask_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Ask LLM with image"""
        if not self.provider.check_service():
            print("LLM service unavailable")
            return ""
        
        if hasattr(self.provider, 'chat_with_image'):
            try:
                return self.provider.chat_with_image(prompt, image_base64, stream)
            except Exception as exc:
                print(f"Vision request failed: {exc}")
                return ""
        else:
            print("Provider does not support image input")
            return ""
    
    @staticmethod
    def extract_code(content: str) -> Optional[str]:
        """从模型回答里抽出 python 代码。兼容三种围栏写法：
        1) 标准  ```python ... ``` ；
        2) qwen2.5-VL 变体  ```python``` ... ```python``` （语言标签和闭合粘在一起，见种子示例被 3b/7b 复述成这种）；
        3) 无语言标签  ``` ... ``` 。
        都抽不出（如只有解释文字）返回 None。"""
        if not content:
            return None
        # 1) 标准围栏
        m = re.search(r"```python(.*?)```", content, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
        # 2) qwen2.5 变体：```python``` 当作起止标记
        m = re.search(r"```python```\s*(.*?)\s*```python```", content, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
        # 3) 无语言标签围栏（要求围栏后换行，避免从 ```python``` 标记里误抓 "python"）
        m = re.search(r"```\s*\n(.*?)\n\s*```", content, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return None
    
    def process(self, command: str, mode: str = "flight") -> Optional[str]:
        response = self.ask(command, mode)
        
        if mode == "flight":
            return self.extract_code(response)
        else:
            return response
    
    def clear_history(self):
        self.chat_history = [self.chat_history[0]]
