"""冒烟验证：加载微调（已 merge）后的 Qwen3-VL-4B，对一张图 + 目标短语做 grounding，
打印模型原始输出并解析 bbox_2d，换算成像素中心。

用法（在装了 ms-swift / transformers>=4.57 的云端环境里）::

    python infer_test.py <模型目录或id> <图片路径> [目标短语] [宽] [高]

示例::

    python infer_test.py output/qwen3vl-4b-refcoco-lora/vx-xxx/merged some.jpg "red ball" 720 480

说明：qwen3-vl 输出为 0-1000 归一化坐标。像素 = 归一化/1000 * (宽,高)。
把本地无人机截图上传来喂它，即可粗略判断「微调后能否锁对目标」。
"""
import argparse
import json
import re
import sys


def parse_bbox(text: str):
    """从模型输出里找第一个 bbox_2d: [x1,y1,x2,y2]，返回 (x1,y1,x2,y2) 归一化坐标或 None。"""
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
                  text)
    if m:
        return tuple(float(g) for g in m.groups())
    # 兜底：直接找第一个 4 元数字组
    m = re.search(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]", text)
    if m:
        return tuple(float(g) for g in m.groups())
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="merge 后的模型目录（如 output/.../merged）或 model id")
    ap.add_argument("image", help="图片路径")
    ap.add_argument("target", nargs="?", default="object", help="要找的目标短语，如 red ball")
    ap.add_argument("width", type=int, nargs="?", default=720)
    ap.add_argument("height", type=int, nargs="?", default=480)
    args = ap.parse_args()

    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    iw, ih = img.size
    w = args.width if args.width else iw
    h = args.height if args.height else ih
    print(f"==> 图片实际 {iw}x{ih}，坐标换算用 {w}x{h}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"==> 加载模型 {args.model} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)

    prompt = (
        f"Find the {args.target} in the image. "
        "Output ONLY a JSON array, e.g. "
        f'[{{"bbox_2d": [x1,y1,x2,y2], "label": "{args.target}"}}]. '
        "If not found, output []."
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    print("==> 生成中 ...")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print("==> 模型原始输出:\n" + raw)

    box = parse_bbox(raw)
    if box is None:
        if "[]" in raw or raw.strip().lower().startswith("[]"):
            print("==> 模型报告：未找到目标（输出 []）")
        else:
            print("==> [!] 没解析到 bbox_2d，输出格式不符合预期 —— 微调/模板可能有问题。")
        return 2

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / 1000.0 * w
    cy = (y1 + y2) / 2.0 / 1000.0 * h
    print(f"==> bbox_2d(归一化): [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    print(f"==> 目标中心像素 (约 {w}x{h}): ({cx:.0f}, {cy:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
